"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""
import dataclasses
from functools import partial
import os
import shutil
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import wandb
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
import numpy as np
import orbax.checkpoint as ocp
import wandb
import functools
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf

from rl.config.config import Config
from rl.model import ActorRNN, CriticRNN, ScannedRNN
from utils import RunnerState, batchify, init_config, init_run, make_sim_render_episode, render_callback, restore_run, save_checkpoint, unbatchify

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    control_mask: jnp.ndarray 

def convert_list_to_pytree(list_of_scenarios):
    """Convert list of scenarios to pytree object."""
    # Flatten each tree
    flat_trees_treedefs = [jax.tree.flatten(tree) for tree in list_of_scenarios]
    flat_trees, treedefs = zip(*flat_trees_treedefs)
    # Concatenate the flattened lists
    concatenated_leaves = [jnp.stack(leaves) for leaves in zip(*flat_trees)]
    # Rebuild PyTree
    return jax.tree.unflatten(treedefs[0], concatenated_leaves)

def sample_from_pytree_callback(scenario_py_tree, idx):
    """Sample a batch of scenes."""
    
    # Select scenario from pytree
    scene_batch_pytree = jax.tree.map(lambda x: x[idx], scenario_py_tree)
    
    return scene_batch_pytree

def make_train(
    config, 
    checkpoint_manager,
    actor_network, 
    env,  
    scene_pytree, 
    batch_size,
    latest_update_step, 
    wandb_run_id,
    scenario,
    sample_scene
):
    # Define which parts of the callback we don't want to trace
    _render_callback = partial(render_callback, save_dir=config._vid_dir)
    _sample_from_pytree_callback = partial(sample_from_pytree_callback, scenario_py_tree=scene_pytree)

    def train(rng, runner_state=None):

        # INIT ENV
        rng, _rng = jax.random.split(rng)

        # INIT NETWORK
        critic_network = CriticRNN(config=config)
        
        jit_sim_render_episode = make_sim_render_episode(config, actor_network, env)
        num_render_actors = 1 * env.num_agents
        ac_init_hstate_render = ScannedRNN.initialize_carry(num_render_actors, config.HIDDEN_DIM)
        render_states = jit_sim_render_episode(runner_state.train_states[0].params, ac_init_hstate_render, scenario)
                
        # DEFINE CALLBACKS
        jax.experimental.io_callback(callback=_render_callback, result_shape_dtypes=None, states=render_states, t=0)
        
        # TRAIN LOOP
        def _update_step_with_render(update_runner_state, unused, render_states):
            
            rng, _ = jax.random.split(update_runner_state[0].rng)
            rand_scenario_idx = jax.random.randint(rng, (1,), 0, config.TRAIN_ON_K_SCENES)[0]
            scenario = jax.experimental.io_callback(
                callback=_sample_from_pytree_callback, 
                result_shape_dtypes=jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), sample_scene),
                idx=rand_scenario_idx,
            )
            
            jax.debug.print("@ _update_step_with_render: using scene {}", scenario.object_metadata.ids.sum())
        
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            
            def _env_step(runner_state: RunnerState, unused):
                
                train_states, env_state, last_obs, last_done, hstates, rng = (
                    runner_state.train_states,
                    runner_state.env_state,
                    runner_state.last_obs,
                    runner_state.last_done,
                    runner_state.hstates, 
                    runner_state.rng
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config._num_actors)
                )
                control_mask = jax.vmap(env._get_control_mask)(env_state.env_state)
                
                obs_batch = batchify(last_obs, env.agents, config._num_actors)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
    
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config.NUM_ENVS, env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0, 1)  
                # shape: (num_envs * num_max_objects, world_state_size)
                world_state = world_state.reshape((config._num_actors, -1))
            
                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, env_act, scenario)
                info = jax.tree.map(lambda x: x.reshape((config._num_actors)), info)
                done_batch = batchify(done, env.agents, config._num_actors).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config._num_actors).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                    avail_actions,
                    control_mask,
                )
                runner_state = RunnerState(train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state.hstates
            
            # DO ROLLOUTS
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
      
            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, rng = (
                runner_state.train_states,
                runner_state.env_state,
                runner_state.last_obs,
                runner_state.last_done,
                runner_state.hstates, 
                runner_state.rng, 
            )

            last_world_state = last_obs["world_state"].swapaxes(0,1)
            last_world_state = last_world_state.reshape((config._num_actors,-1))
            
            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                                
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value
            
            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info
                    
                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
              
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        
                        # Average
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        
                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)
                    
                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        # RERUN NETWORK
                        _, value = critic_network.apply(critic_params, init_hstate.squeeze(), (traj_batch.world_state,  traj_batch.done)) 
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)
                    
                    # COMPUTE ACTOR AND CRITIC GRADIENTS
                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )
                    
                    # INSERT GRADIENTS
                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    
                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }
                    
                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(lambda x: jnp.reshape(
                    x, (1, config._num_actors, -1)
                ), init_hstates)
                
                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config._num_actors)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                
                train_states, loss_info = jax.lax.scan(
                    _update_minibatch, train_states, minibatches
                )
                      
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info
            
            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            
            train_states = update_state[0]
            metric = traj_batch.info
          
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            metric["loss"] = loss_info
     
            metric["obs_dist_min"] = traj_batch.obs.min()
            metric["obs_dist_max"] = traj_batch.obs.max()
            metric["obs_dist_std"] = traj_batch.obs.std()
            metric["act_dist_mean"] = traj_batch.action.mean()
            metric["act_dist_std"] = traj_batch.action.std()
            
            rng = update_state[-1]
            
            def callback(metric):

                wandb.log(
                    {
                        # the metrics have an agent dimension, but this is identical
                        # for all agents so index into the 0th item of that dimension.
                        "returns": metric["returned_episode_returns"][:, :, 0][
                            metric["returned_episode"][:, :, 0]
                        ].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                        **metric["loss"],
                        "obs_dist_min": metric["obs_dist_min"],
                        "obs_dist_max": metric["obs_dist_max"],
                        "obs_dist_std": metric["obs_dist_std"],
                        "act_dist_mean": metric["act_dist_mean"],
                        "act_dist_std": metric["act_dist_std"],
                    }
                )

            def ckpt_callback(metric, runner_state):
                try:
                    curr_update_step = metric["update_steps"]
                    save_checkpoint(config, checkpoint_manager, runner_state, curr_update_step)
                except jax.errors.ConcretizationTypeError:
                    return
            
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = RunnerState(train_states, env_state, last_obs, last_done, hstates, rng)
            do_render = update_steps % config.RENDER_FREQ == 0 
            
            render_states = jax.lax.cond(
                do_render,
                partial(jit_sim_render_episode, runner_state.train_states[0].params, ac_init_hstate_render, scenario=scenario),
                lambda: render_states,
            )
            jax.lax.cond(
                do_render,
                partial(jax.experimental.io_callback, _render_callback, None, states=render_states, t=update_steps),
                lambda: None,
            )
            jax.lax.cond(
                update_steps % config.CKPT_FREQ == 0,
                partial(jax.experimental.io_callback, ckpt_callback, None, metric, runner_state),
                lambda: None,
            )
                        
            return (runner_state, update_steps), metric
        
        
        _update_step = functools.partial(_update_step_with_render, render_states=render_states)
        
        runner_state, metric = jax.lax.scan(
            _update_step,   
            (runner_state, latest_update_step), None, config._num_updates - latest_update_step
        )
        
        return {"runner_state": runner_state} 

    return train
    
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: Config):
    init_config(config)
    
    if config.OVERWRITE:
        shutil.rmtree(config._exp_dir, ignore_errors=True)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        config._ckpt_DIR, options=options)

    rng = jax.random.PRNGKey(config.SEED)
    latest_update_step = checkpoint_manager.latest_step()
    
    runner_state, actor_network, env, scenario, latest_update_step, data_iter = \
        init_run(config, checkpoint_manager, latest_update_step, rng)
    
    # Create pytree with scenarios
    list_of_scenarios = []
    for _ in range(config.TRAIN_ON_K_SCENES):
        list_of_scenarios.append(next(data_iter))
        
    scenario_pytree = convert_list_to_pytree(list_of_scenarios)
    
    # Transfer pytree to cpu memory (otherwise we go OOM)
    jax.device_put(x=scenario_pytree, device=jax.devices("cpu")[0])
            
    if latest_update_step is not None:
        runner_state, wandb_run_id = restore_run(config, runner_state, checkpoint_manager, latest_update_step)
        wandb_resume = "Must"
    else:
        wandb_run_id, wandb_resume = None, None

    latest_update_step = 0 if latest_update_step is None else latest_update_step

    os.makedirs(config._exp_dir, exist_ok=True)
    os.makedirs(config._vid_dir, exist_ok=True)

    run = wandb.init(
        project=config.PROJECT,
        tags=["MAPPO"],
        config=OmegaConf.to_container(config),
        mode=config.WANDB_MODE,
        dir=config._exp_dir,
        id=wandb_run_id,
        resume=wandb_resume,
    )
    wandb_run_id = run.id
    with open(os.path.join(config._exp_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb_run_id)
    
    with jax.disable_jit(False):
        
        print(f'@ jit: using scene {scenario.object_metadata.ids.sum()}')
        
        sample_scene = sample_from_pytree_callback(scenario_pytree, idx=0)
        
        train_jit = jax.jit(
            make_train(
                config=config, 
                checkpoint_manager=checkpoint_manager, 
                env=env, 
                actor_network=actor_network,
                latest_update_step=latest_update_step, 
                wandb_run_id=run.id,
                scenario=scenario, 
                batch_size=jnp.array((1)),
                scene_pytree=scenario_pytree,
                sample_scene=sample_scene,
            )
        ) 
        
        out = train_jit(rng, runner_state=runner_state)

    runner_state = out["runner_state"]
    n_updates = runner_state[-1]
    runner_state: RunnerState = runner_state[0]

if __name__=="__main__":
    main()
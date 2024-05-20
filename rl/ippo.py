"""
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import wandb
import dataclasses
import functools
import matplotlib.pyplot as plt
from functools import partial
import hydra
import pickle
from omegaconf import OmegaConf
from utils import batchify, init_config

# WAYMAX
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import agents
from waymax import visualization
from waymax.config import LinearCombinationRewardConfig
from waymax.datatypes import operations, observation, roadgraph
from waymax.datatypes.simulator_state import SimulatorState

# Helpers
from rl.wrappers.marl_ippo import WaymaxLogEnvState, WaymaxLogWrapper, WaymaxWrapper

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    
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

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(
    config,
    env,
    scenario,
    scenario_tree,
):
    # DEFINE CALLBACKS
    _sample_from_pytree_callback = partial(
        sample_from_pytree_callback, 
        scenario_py_tree=scenario_tree
    )
        
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
     
        # INIT NETWORK
        network = ActorCritic(env.action_space(env.agents[0]).shape[0], activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

        # Create train state object
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(None, 0))(scenario, reset_rng)

        # TRAIN LOOP
        def _update_step_with_render(update_runner_state, unused, scenario):
                    
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            
            _, _, _, rng = runner_state
            
            # SAMPLE NEW SCENARIO
            rand_scenario_idx = jax.random.randint(rng, (1,), 0, config.TRAIN_ON_K_SCENES)[0]
            scenario = jax.experimental.io_callback(
                callback=_sample_from_pytree_callback, 
                result_shape_dtypes=jax.tree.map(
                    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), 
                    scenario,
                ),
                idx=rand_scenario_idx,
            )
            
            jax.debug.print(
                "@ _update_step_with_render: using scene {}", 
                scenario.object_metadata.ids.sum()
            )
        
            def _env_step(runner_state, unused):
                
                train_state, env_state, last_obs, rng = runner_state

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, env_act, scenario)

                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                
                
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action,
                    value.squeeze(),
                    batchify(reward, env.agents, config._num_actors).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            avail_actions = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).shape[0])
            )
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)
            _, last_val = network.apply(train_state.params, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
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
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params,
                                                  (traj_batch.obs, traj_batch.done, traj_batch.avail_actions))
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
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
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

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

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (
                train_state, 
                traj_batch, 
                advantages, 
                targets, 
                rng,
            )
            
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric):
                
                jax.debug.breakpoint()
                
                # Compute average across agents
                # Get ~ average return per episode (episode length is 80 steps)
                returns =  metric["returned_episode_returns"].sum() / (80 * config.NUM_ENVS * 10)
                
                wandb.log(
                    {
                        "returns": returns.item(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                    }
                )
                
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, rng)
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        
        _update_step = functools.partial(_update_step_with_render, scenario=scenario)
        
        jax.debug.breakpoint()
        
        runner_state = (train_state, env_state, obsv, _rng)
        
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )    
    
        return {"runner_state": runner_state}

    return train

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    init_config(config)
    
    # CONSTANTS
    MAX_NUM_OBJECTS = 10
    
    # CREATE DATA ITERATOR
    data_config = dataclasses.replace(
    _config.WOD_1_0_0_TRAINING,
        max_num_objects=MAX_NUM_OBJECTS,
    )
    data_iter = dataloader.simulator_state_generator(config=data_config)
    #scenario = next(data_iter) # Get a scenario 
    
    scenario_name = f"scenario_{MAX_NUM_OBJECTS}-max-objects.pkl"
    # with open(scenario_name, "wb") as f:
    #     pickle.dump(scenario, f) 
        
    with open(scenario_name, "rb") as f:
        scenario: SimulatorState = pickle.load(f) 
        
     # Create pytree with scenarios        
    scenario_pytree = convert_list_to_pytree([scenario])
        
    
    # MAKE AND WRAP ENVIRONMENT
    dynamics_model = dynamics.InvertibleBicycleModel(
        normalize_actions=True,  # This means we feed in all actions as in [-1, 1]
    )
    
    # Use relative coordinates
    obs_config = dataclasses.replace(
        _config.ObservationConfig(), 
        coordinate_frame=_config.CoordinateFrame.OBJECT,
        roadgraph_top_k=config.TOPK_ROADPOINTS,
    )

    # Create env config
    env_config = dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=MAX_NUM_OBJECTS,
        observation=obs_config,
        rewards=_config.LinearCombinationRewardConfig(
            {
                'offroad': config.OFFROAD,
                'overlap': config.OVERLAP, 
                'log_divergence': config.LOG_DIVERGENCE,
            }
        ),
        
        # Controll all valid objects in the scene.
        controlled_object=_config.ObjectType.VALID,
    )

    # Create waymax environment
    waymax_base_env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=env_config,
    )
    
    # Wrap environment with JAXMARL wrapper
    env = WaymaxWrapper(
        waymax_base_env, 
        obs_with_agent_id=False
    )

    # Wrap environment with LogWrapper
    env = WaymaxLogWrapper(env)
   
    # Initialize wandb
    wandb.init(
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=OmegaConf.to_container(config) ,
    )

    rng = jax.random.PRNGKey(50)
   
    scenario = sample_from_pytree_callback(scenario_pytree, idx=0)
    
    with jax.disable_jit(False):
        # TRAIN
        train_jit = jax.jit(
            make_train(
                config=config, 
                env=env, 
                scenario=scenario,
                scenario_tree=scenario_pytree,
            ), 
            device=jax.devices()[0],
        )
        out = train_jit(rng)


if __name__ == "__main__":
    main()
    
    
    '''results = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
    jnp.save('hanabi_results', results)
    plt.plot(results)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'IPPO_{config["ENV_NAME"]}.png')'''
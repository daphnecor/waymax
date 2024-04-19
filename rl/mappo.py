"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""
import dataclasses
from functools import partial
import os
import shutil
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict

import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
# from flax.training import orbax_utils
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
import functools
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from time import perf_counter

from rl.environments.spaces import Box
from rl.environments.multi_agent_env import MultiAgentEnv
from rl.wrappers.baselines import JaxMARLWrapper, LogWrapper, WaymaxLogWrapper
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import agents
from waymax import visualization
from waymax.datatypes.action import Action
from waymax.datatypes.simulator_state import SimulatorState
from rl.wrappers.marl import WaymaxWrapper


@struct.dataclass
class RunnerState:
    train_states: Tuple[TrainState, TrainState]
    env_state: MultiAgentEnv
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    hstates: Tuple[jnp.ndarray, jnp.ndarray]
    rng: jnp.ndarray


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        #print('ins', ins)
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions

        # action_logits = actor_mean - (unavail_actions * 1e10)
        # pi = distrax.Categorical(logits=action_logits)
        actor_mean = actor_mean - (unavail_actions * 1e10)
        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, jnp.squeeze(critic, axis=-1)


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


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    #print('batchify', x.shape)
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

    
max_num_objects = 32


def linear_schedule(config, count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac


def make_train(config, checkpoint_manager, env, scenario, latest_update_step, wandb_run_id):
    
    # t_start_up_start = perf_counter()
   
    # # Configure dataset
    # t_data_iter_start = perf_counter()
    # dataset_config = dataclasses.replace(_config.WOD_1_0_0_VALIDATION, max_num_objects=max_num_objects)
    # data_iter = dataloader.simulator_state_generator(config=dataset_config)
    
    # t_data_iter_end = perf_counter()
    
    # t_next_start = perf_counter()
    
    # scenario = next(data_iter)
    
    # t_next_end = perf_counter() 
    
    # dynamics_model = dynamics.InvertibleBicycleModel()

    # # Create waymax environment
    # waymax_base_env = _env.MultiAgentEnvironment(
    #     dynamics_model=dynamics_model,
    #     config=dataclasses.replace(
    #         _config.EnvironmentConfig(),
    #         max_num_objects=max_num_objects,
    #         controlled_object=_config.ObjectType.VALID,
    #     ),
    # )
    # # Wrap environment with JAXMARL wrapper
    # env = WaymaxWrapper(waymax_base_env, obs_with_agent_id=False)

    # # Wrap environment with LogWrapper
    # env = WaymaxLogWrapper(env)

    # # Configure training
    # config.NUM_ACTORS = env.num_agents * config.NUM_ENVS
    # config.NUM_UPDATES = (
    #     config["TOTAL_TIMESTEPS"] // config.NUM_STEPS // config["NUM_ENVS"]
    # )
    # config.MINIBATCH_SIZE = (
    #     config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    # )
    # config["CLIP_EPS"] = (
    #     config["CLIP_EPS"] / env.num_agents
    #     if config["SCALE_CLIP_EPS"]
    #     else config["CLIP_EPS"]
    # )
 
    # t_start_up_end = perf_counter()
    
    # print(f"--- TOTAL STARTUP COSTS (make data_iter + env) = {t_start_up_end - t_start_up_start} s ---")
    # print(f" of which \n")
    # print(f"--- DATA ITER COSTS = {t_data_iter_end - t_data_iter_start} s ---")
    # print(f"--- NEXT DATA ITER COSTS = {t_next_end - t_next_start} s ---")
    

    def train(rng, runner_state=None):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        # reset_rng = jax.random.split(_rng, config.NUM_ENVS)
        # obsv, env_state = jax.vmap(env.reset, in_axes=(None, 0))(scenario, reset_rng)
        # ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], 128)
        # cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], 128)

        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).shape[0], config=config)
        critic_network = CriticRNN(config=config)

        # if runner_state is None:
        #     rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        #     ac_init_x = (
        #         jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
        #         jnp.zeros((1, config["NUM_ENVS"])),
        #         jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).shape[0])),
        #     )
        #     ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        #     actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        #     cr_init_x = (
        #         jnp.zeros((1, config["NUM_ENVS"], env.world_state_size(),)),  
        #         jnp.zeros((1, config["NUM_ENVS"])),
        #     )
        #     cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        #     critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        #     _linear_schedule = partial(linear_schedule, config)
            
        #     if config["ANNEAL_LR"]:
        #         actor_tx = optax.chain(
        #             optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        #             optax.adam(learning_rate=_linear_schedule, eps=1e-5),
        #         )
        #         critic_tx = optax.chain(
        #             optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        #             optax.adam(learning_rate=_linear_schedule, eps=1e-5),
        #         )
        #     else:
        #         actor_tx = optax.chain(
        #             optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        #             optax.adam(config["LR"], eps=1e-5),
        #         )
        #         critic_tx = optax.chain(
        #             optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        #             optax.adam(config["LR"], eps=1e-5),
        #         )
        #     actor_train_state = TrainState.create(
        #         apply_fn=actor_network.apply,
        #         params=actor_network_params,
        #         tx=actor_tx,
        #     )
        #     critic_train_state = TrainState.create(
        #         apply_fn=actor_network.apply,
        #         params=critic_network_params,
        #         tx=critic_tx,
        #     )


        #     rng, _rng = jax.random.split(rng)
        #     runner_state = RunnerState(
        #         (actor_train_state, critic_train_state),
        #         env_state,
        #         obsv,
        #         jnp.zeros((config.NUM_ACTORS), dtype=bool, ),
        #         (ac_init_hstate, cr_init_hstate),
        #         _rng,
        #     )

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            
            def _env_step(runner_state: RunnerState, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = (runner_state.train_states,
                                                                              runner_state.env_state,
                                                                              runner_state.last_obs,
                                                                              runner_state.last_done,
                                                                              runner_state.hstates, runner_state.rng)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
                #print('env step ac in', ac_in)
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0,1)  
                world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                
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
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                    avail_actions,
                )
                runner_state = RunnerState(train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state.hstates
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, rng = (runner_state.train_states,
                                                                            runner_state.env_state,
                                                                            runner_state.last_obs,
                                                                            runner_state.last_done,
                                                                            runner_state.hstates, runner_state.rng)
            
            last_world_state = last_obs["world_state"].swapaxes(0,1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            
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

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )
                    
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
                    x, (1, config["NUM_ACTORS"], -1)
                ), init_hstates)
                
                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
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

                #train_states = (actor_train_state, critic_train_state)
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
            rng = update_state[-1]

            def callback(metric):
                
                # Create a video
                # frames = env.render(scenario, waymax_base_env, dynamics_model)
                # Video must be atleast 4 dimensions: time, channels, height, width

                wandb.log(
                    {
                        # the metrics have an agent dimension, but this is identical
                        # for all agents so index into the 0th item of that dimension.
                        "returns": metric["returned_episode_returns"][:, :, 0][
                            metric["returned_episode"][:, :, 0]
                        ].mean(),
                        # "win_rate": metric["returned_won_episode"][:, :, 0][
                        #     metric["returned_episode"][:, :, 0]
                        # ].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                        **metric["loss"],
                        # "video": wandb.Video(frames, fps=10, format="gif"),
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
            jax.lax.cond(
                update_steps % config.ckpt_freq == 0,
                partial(jax.experimental.io_callback, ckpt_callback, None, metric, runner_state),
                lambda: None,
            )
            return (runner_state, update_steps), metric

        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, latest_update_step), None, config.NUM_UPDATES - latest_update_step
        )
        return {"runner_state": runner_state}

    return train

    
def get_exp_dir(config):
    exp_dir = os.path.join(
        'saves',
        f"{config.SEED}"
    )
    return exp_dir


def get_ckpt_dir(config):
    ckpts_dir = os.path.abspath(os.path.join(config.exp_dir, "ckpts"))
    return ckpts_dir

    
def init_config(config):
    config.exp_dir = get_exp_dir(config)
    config.ckpt_dir = get_ckpt_dir(config)

    
def save_checkpoint(config, ckpt_manager, runner_state, t):
    ckpt_manager.save(t.item(), args=ocp.args.StandardSave(runner_state))
    ckpt_manager.wait_until_finished()

    
def init_or_restore_run(config, ckpt_manager, latest_update_step, rng):
    t_start_up_start = perf_counter()
   
    # Configure dataset
    t_data_iter_start = perf_counter()
    dataset_config = dataclasses.replace(_config.WOD_1_0_0_VALIDATION, max_num_objects=max_num_objects)
    data_iter = dataloader.simulator_state_generator(config=dataset_config)
    
    t_data_iter_end = perf_counter()
    
    t_next_start = perf_counter()
    
    scenario = next(data_iter)
    
    t_next_end = perf_counter() 
    
    dynamics_model = dynamics.InvertibleBicycleModel()

    # Create waymax environment
    waymax_base_env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(
            _config.EnvironmentConfig(),
            max_num_objects=max_num_objects,
            controlled_object=_config.ObjectType.VALID,
        ),
    )
    # Wrap environment with JAXMARL wrapper
    env = WaymaxWrapper(waymax_base_env, obs_with_agent_id=False)

    # Wrap environment with LogWrapper
    env = WaymaxLogWrapper(env)

    # Configure training
    config.NUM_ACTORS = env.num_agents * config.NUM_ENVS
    config.NUM_UPDATES = (
        config["TOTAL_TIMESTEPS"] // config.NUM_STEPS // config["NUM_ENVS"]
    )
    config.MINIBATCH_SIZE = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )
 
    t_start_up_end = perf_counter()
    
    print(f"--- TOTAL STARTUP COSTS (make data_iter + env) = {t_start_up_end - t_start_up_start} s ---")
    print(f" of which \n")
    print(f"--- DATA ITER COSTS = {t_data_iter_end - t_data_iter_start} s ---")
    print(f"--- NEXT DATA ITER COSTS = {t_next_end - t_next_start} s ---")

    actor_network = ActorRNN(env.action_space(env.agents[0]).shape[0], config=config)
    critic_network = CriticRNN(config=config)
    rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
    ac_init_x = (
        jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
        jnp.zeros((1, config["NUM_ENVS"])),
        jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).shape[0])),
    )
    ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
    cr_init_x = (
        jnp.zeros((1, config["NUM_ENVS"], env.world_state_size(),)),  
        jnp.zeros((1, config["NUM_ENVS"])),
    )
    cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

    _linear_schedule = partial(linear_schedule, config)
    
    if config["ANNEAL_LR"]:
        actor_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=_linear_schedule, eps=1e-5),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=_linear_schedule, eps=1e-5),
        )
    else:
        actor_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
    actor_train_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_network_params,
        tx=actor_tx,
    )
    critic_train_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=critic_network_params,
        tx=critic_tx,
    )

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.NUM_ENVS)
    obsv, env_state = jax.vmap(env.reset, in_axes=(None, 0))(scenario, reset_rng)
    ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], 128)
    cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], 128)

    rng, _rng = jax.random.split(rng)
    runner_state = RunnerState(
        (actor_train_state, critic_train_state),
        env_state,
        obsv,
        jnp.zeros((config.NUM_ACTORS), dtype=bool, ),
        (ac_init_hstate, cr_init_hstate),
        _rng,
    )

    if ckpt_manager.latest_step() is not None:
        runner_state = ckpt_manager.restore(latest_update_step, args=ocp.args.StandardRestore(runner_state))
        wandb_resume = 'Must'
        with open(os.path.join(config.exp_dir, "wandb_run_id.txt"), "r") as f:
            wandb_run_id = f.read()
    else:
        wandb_resume = None
        wandb_run_id = None

    return runner_state, env, scenario, latest_update_step, wandb_run_id, wandb_resume
    


@hydra.main(version_base=None, config_path="config", config_name="mappo_homogenous_rnn_waymax")
def main(config):
    init_config(config)
    if config.overwrite:
        shutil.rmtree(config.exp_dir, ignore_errors=True)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        config.ckpt_dir, options=options)

    rng = jax.random.PRNGKey(config.SEED)
    latest_update_step = checkpoint_manager.latest_step()
    runner_state, env, scenario, latest_update_step, wandb_run_id, wandb_resume = init_or_restore_run(config, checkpoint_manager, latest_update_step, rng)
    latest_update_step = 0 if latest_update_step is None else latest_update_step

    os.makedirs(config.exp_dir, exist_ok=True)

    run = wandb.init(
        # entity=config.ENTITY,
        project=config.PROJECT,
        tags=["MAPPO", config.MAP_NAME],
        config=OmegaConf.to_container(config),
        mode=config.WANDB_MODE,
        dir=config.exp_dir,
        id=wandb_run_id,
        resume=wandb_resume,
    )
    wandb_run_id = run.id
    with open(os.path.join(config.exp_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb_run_id)
    with jax.disable_jit(False):
        train_jit = jax.jit(make_train(config, checkpoint_manager, env=env, scenario=scenario,
                                       latest_update_step=latest_update_step, wandb_run_id=run.id)) 
        out = train_jit(rng, runner_state=runner_state)

    runner_state = out["runner_state"]
    n_updates = runner_state[-1]
    runner_state: RunnerState = runner_state[0]

    
if __name__=="__main__":
    main()
import dataclasses
from functools import partial
import os
import pickle
import shutil
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict

import chex
import imageio
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
import hydra
from omegaconf import DictConfig, OmegaConf
from time import perf_counter

from rl.config.config import Config
from rl.environments.spaces import Box
from rl.environments.multi_agent_env import MultiAgentEnv
from rl.model import ActorRNN, CriticRNN, ScannedRNN
from rl.wrappers.baselines import JaxMARLWrapper, LogWrapper
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import agents
from waymax import visualization
from waymax.datatypes.action import Action
from waymax.datatypes.simulator_state import SimulatorState
from rl.wrappers.marl import WaymaxLogEnvState, WaymaxLogWrapper, WaymaxWrapper


@struct.dataclass
class RunnerState:
    train_states: Tuple[TrainState, TrainState]
    env_state: MultiAgentEnv
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    hstates: Tuple[jnp.ndarray, jnp.ndarray]
    rng: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    #print('batchify', x.shape)
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def linear_schedule(config, count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac

    
# Reload single scenario from disk for speedier debugging
DEBUG_WITH_ONE_SCENARIO = True


def init_run(config: Config, ckpt_manager, latest_update_step, rng):
    t_start_up_start = perf_counter()
    
    # HACK when debugging with one scenario to avoid long loading time for data iterator
    scenario_name = f"scenario_{config.max_num_objects}-max-objects.pkl"
    if DEBUG_WITH_ONE_SCENARIO and os.path.isfile(scenario_name):
        t_data_iter_start = perf_counter()
        t_next_start = perf_counter()
        with open(scenario_name, "rb") as f:
           scenario: SimulatorState = pickle.load(f) 
        t_data_iter_end = perf_counter()
        t_next_end = perf_counter() 

    else:
        # Configure dataset
        t_data_iter_start = perf_counter()
        dataset_config = dataclasses.replace(_config.WOD_1_0_0_VALIDATION, max_num_objects=config.max_num_objects)
        data_iter = dataloader.simulator_state_generator(config=dataset_config)
        t_data_iter_end = perf_counter()
    
        t_next_start = perf_counter()
        scenario: SimulatorState = next(data_iter)
        t_next_end = perf_counter() 
        # Save this scenario to disk for later use
        with open(scenario_name, "wb") as f:
            pickle.dump(scenario, f)
    
    dynamics_model = dynamics.InvertibleBicycleModel()

    # Create env config
    env_config = dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=config.max_num_objects,
        rewards=_config.LinearCombinationRewardConfig(
            {
                # 'overlap': -1.0, 
                # 'offroad': -1.0, 
                'log_divergence': 1.0
            }
        ),
        #metrics={}
        # Controll all valid objects in the scene.
        controlled_object=_config.ObjectType.VALID,
    )

    # Create waymax environment
    waymax_base_env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=env_config,
    )
    # Wrap environment with JAXMARL wrapper
    env = WaymaxWrapper(waymax_base_env, obs_with_agent_id=False)

    # Wrap environment with LogWrapper
    env = WaymaxLogWrapper(env)

    # Configure training
    config.NUM_ACTORS = env.num_agents * config.NUM_ENVS
    config.NUM_UPDATES = int(
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
    ac_init_hstate = ScannedRNN.initialize_carry(config.NUM_ENVS, config["GRU_HIDDEN_DIM"])
    actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
    cr_init_x = (
        jnp.zeros((1, config["NUM_ENVS"], env.world_state_size(),)),  
        jnp.zeros((1, config["NUM_ENVS"])),
    )
    cr_init_hstate = ScannedRNN.initialize_carry(config.NUM_ENVS, config["GRU_HIDDEN_DIM"])
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

    return runner_state, env, scenario, latest_update_step


def restore_run(config: Config, runner_state: RunnerState, ckpt_manager, latest_update_step: int):
    if latest_update_step is not None:
        runner_state = ckpt_manager.restore(latest_update_step, args=ocp.args.StandardRestore(runner_state))
        with open(os.path.join(config.exp_dir, "wandb_run_id.txt"), "r") as f:
            wandb_run_id = f.read()

    return runner_state, wandb_run_id


def make_sim_render_episode(config: Config, actor_network, env, scenario):
    # FIXME: Shouldn't hardcode this
    max_episode_len = 91

    rng = jax.random.PRNGKey(0)
    init_obs, init_state = env.reset(scenario, rng)
    init_obs = batchify(init_obs, env.agents, env.num_agents)
    # remaining_timesteps = init_state.env_state.remaining_timesteps
    # actor_params = runner_state.train_states[0].params
    # actor_hidden = runner_state.hstates[0]

    def sim_render_episode(actor_params, actor_hidden):
        def step_env(carry, _):
            rng, obs, state, done, actor_hidden = carry
            # print(obs.shape)

            # traj = datatypes.dynamic_index(
            #     state.env_state.sim_trajectory, state.env_state.timestep, axis=-1, keepdims=True
            # )
            avail_actions = env.get_avail_actions(state.env_state)
            avail_actions = jax.lax.stop_gradient(
                batchify(avail_actions, env.agents, len(env.agents))
            )
            ac_in = (
                obs[np.newaxis, :],
                # obs,
                done[np.newaxis, :],
                # done,
                avail_actions[np.newaxis, :],
            )
            actor_hidden, pi = actor_network.apply(actor_params, actor_hidden, ac_in)            
            action = pi.sample(seed=rng)
            env_act = unbatchify(
                action, env.agents, 1, env.num_agents
            )
            env_act = {k: v.squeeze() for k, v in env_act.items()}

            # outputs = [
            #     jit_select_action({}, state, obs, None, rng)
            #     for jit_select_action in jit_select_action_list
            # ]
            # action = agents.merge_actions(outputs)
            obs, next_state, reward, done, info = env.step(state=state, action=env_act, scenario=scenario, key=rng)
            rng, _ = jax.random.split(rng)
            done = batchify(done, env.agents, env.num_agents)[:, 0]
            obs = batchify(obs, env.agents, env.num_agents)

            return (rng, obs, next_state, done, actor_hidden), next_state

            
        done = jnp.zeros((len(env.agents),), dtype=bool)

        _, states = jax.lax.scan(step_env, (rng, init_obs, init_state, done, actor_hidden), None, length=max_episode_len)

        # Concatenate the init_state to the states
        states = jax.tree.map(lambda x, y: jnp.concatenate([x[None], y], axis=0), init_state, states)

        return states

    return jax.jit(sim_render_episode)

# states = []
# rng, obs, state, done, actor_hidden = (rng, init_obs, init_state, done, actor_hidden)
# for i in range(remaining_timesteps):
#     carry, state = step_env((rng, obs, state, done, actor_hidden), None)
#     rng, obs, state, done, actor_hidden = carry
#     states.append(state)

    
def render_callback(states: WaymaxLogEnvState, save_dir: str, t: int):

    frames = []
    for i in range(states.env_state.remaining_timesteps[0].item()):
        if (i == 0) or ((i + 1) % 5 == 0):
            # state = jax.tree.map(lambda x: x[i] if len(x.shape) > 0 else x, states)
            state = jax.tree.map(lambda x: x[i], states)
            state = jax.device_put(state, jax.devices('cpu')[0])
            with jax.disable_jit():
                frames.append(visualization.plot_simulator_state(state.env_state, use_log_traj=False,
                                                                 render_overlaps=False))

    imageio.mimsave(os.path.join(save_dir, f"enjoy_{t}.gif"), frames, fps=10, loop=0)
    wandb.log({"video": wandb.Video(os.path.join(save_dir, f"enjoy_{t}.gif"), fps=10, format="gif")})


def get_exp_dir(config):
    exp_dir = os.path.join(
        'saves',
        f"{config.SEED}"
    )
    return exp_dir


def get_ckpt_dir(config):
    ckpts_dir = os.path.abspath(os.path.join(config.exp_dir, "ckpts"))
    return ckpts_dir

    
def init_config(config: Config):
    config.exp_dir = get_exp_dir(config)
    config.ckpt_dir = get_ckpt_dir(config)
    config.vid_dir = os.path.join(config.exp_dir, "vids")

    
def save_checkpoint(config, ckpt_manager, runner_state, t):
    ckpt_manager.save(t.item(), args=ocp.args.StandardSave(runner_state))
    ckpt_manager.wait_until_finished() 


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
import hydra
from omegaconf import DictConfig, OmegaConf
from time import perf_counter

from rl.config.config import Config
from rl.environments.spaces import Box
from rl.environments.multi_agent_env import MultiAgentEnv
from rl.model import ActorRNN, CriticRNN, ScannedRNN
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


def init_or_restore_run(config: Config, ckpt_manager, latest_update_step, rng):
    t_start_up_start = perf_counter()
   
    # Configure dataset
    t_data_iter_start = perf_counter()
    dataset_config = dataclasses.replace(_config.WOD_1_0_0_VALIDATION, max_num_objects=config.max_num_objects)
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
            max_num_objects=config.max_num_objects,
            controlled_object=_config.ObjectType.VALID,
        ),
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

    if latest_update_step is not None:
        runner_state = ckpt_manager.restore(latest_update_step, args=ocp.args.StandardRestore(runner_state))
        wandb_resume = 'Must'
        with open(os.path.join(config.exp_dir, "wandb_run_id.txt"), "r") as f:
            wandb_run_id = f.read()
    else:
        wandb_resume = None
        wandb_run_id = None

    return runner_state, env, scenario, latest_update_step, wandb_run_id, wandb_resume
"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""
import dataclasses
from functools import partial
import os
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict

import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training import orbax_utils
import numpy as np
import optax
import orbax
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

max_num_objects = 32

class WaymaxWrapper(JaxMARLWrapper):
    """
    Provides a `"world_state"` observation for the centralised critic.
    world state observation of dimension: (num_agents, world_state_size)    
    """
    
    def __init__(self,
                 env: _env.MultiAgentEnvironment,
                 obs_with_agent_id=True,
                 ):
        super().__init__(env)
        self.num_agents = self._env.config.max_num_objects
        self.obs_with_agent_id = obs_with_agent_id
        self.agents = [
            f'object_{i}' for i in range(self.num_agents)
        ]
        
        self._state_size = 6
        self.world_state_fn = self.ws_just_env_state

        if not self.obs_with_agent_id:
            self._world_state_size = self._state_size * self.num_agents
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._state_size * self.num_agents + self.num_agents
            self.world_state_fn = self.ws_with_agent_id

        self.observation_spaces = {
            i: Box(low=-1, high=1.0, shape=(self._state_size * self.num_agents + self.num_agents,)) for i in self.agents
        }
        self.action_spaces = {
            i: Box(low=-1, high=1.0, shape=(2,)) for i in self.agents 
        }

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=0)
    def get_obs(env, env_state):
        traj = datatypes.dynamic_index(
            env_state.sim_trajectory, env_state.timestep, axis=-1, keepdims=True
        )
        world_state = jnp.concatenate(
        (traj.xy, traj.yaw[..., None], traj.vel_x[..., None], traj.vel_y[..., None], traj.vel_yaw[..., None]), axis=-1
        )[:, 0]
        agent_ids = jnp.eye(env.num_agents)
        obs = {
            agent: jnp.concatenate((world_state, agent_ids[i][..., None]), axis=-1)
            for i, agent in enumerate(env.agents)
        }
        obs.update(
            {
                "world_state": world_state[None].repeat(env.num_agents, axis=0)
            }
        )
        return obs
        

    def get_avail_actions(self, env_state):
        """TODO: Add me."""
        return {
            agent: jnp.ones((2,)) for agent in self.agents
        }

    
    @partial(jax.jit, static_argnums=0)
    def reset(self,
              scenario,
              key):
        """TODO: Add me."""
        env_state = self._env.reset(scenario, key)
        obs = self.get_obs(env_state)
        # obs["world_state"] = self.world_state_fn(obs, env_state)
        return obs, env_state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: SimulatorState,
        actions: Dict[str, chex.Array],
        scenario
    ) -> Tuple[Dict[str, chex.Array], SimulatorState, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""

        # key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(state, actions)

        rewards = {agent: rewards[i] for i, agent in enumerate(self.agents)}

        obs_re, states_re = self.reset(scenario, key)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos
    

    @partial(jax.jit, static_argnums=0)
    def step_env(self, state, action):
        data = jnp.stack([action[agent] for agent in self.agents])
        traj = datatypes.dynamic_index(
                state.sim_trajectory, state.timestep, axis=-1, keepdims=True
        )

        # TODO: Is this right?
        valid = traj.valid

        action = Action(data=data, valid=valid)
        reward = self._env.reward(state, action)
        env_state = self._env.step(state, action)
        obs = self.get_obs(env_state)
        info = {}
        obs = self.get_obs(env_state)
        done = {'__all__': env_state.is_done}
        done.update(
            {agent: valid_i for agent, valid_i in zip(self.agents, valid)}
        )
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, state):
        #return all_obs
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state
        
    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        #all_obs = jnp.array([obs[agent] for agent in self._env.agents])
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)
        
    def world_state_size(self):
        return self._world_state_size 
    
    def render(self, scenario, env, dynamics_model, init_steps = 11):
        """Make a video of the policy acting in the environment."""
        
        # IDM actor/policy controlling both object 0 and 1.
        # TODO: use our policy 
        actor = agents.IDMRoutePolicy(
            # Controlled objects are those valid at t=0.
            is_controlled_func=lambda state: state.log_trajectory.valid[..., init_steps]
        )
        
        actors = [actor]

        jit_step = jax.jit(env.step)
        jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

        states = [env.reset(scenario)]
        frames = []
        rng = jax.random.PRNGKey(0)
        for i in range(states[0].remaining_timesteps):
            current_state = states[-1]

            outputs = [
                jit_select_action({}, current_state, None, rng)
                for jit_select_action in jit_select_action_list
            ]
            action = agents.merge_actions(outputs)
            next_state = jit_step(current_state, action)
            rng, _ = jax.random.split(rng)

            states.append(next_state)
            
            if i % 5 == 0:
                frames.append(visualization.plot_simulator_state(next_state, use_log_traj=False))

        return np.array(frames)
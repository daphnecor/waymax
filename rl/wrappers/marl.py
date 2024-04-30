"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""
from functools import partial
import struct
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict

import chex
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
import numpy as np
from flax.training.train_state import TrainState
from flax import struct
import hydra
from omegaconf import DictConfig, OmegaConf
from time import perf_counter

from rl.environments.spaces import Box
from rl.environments.multi_agent_env import MultiAgentEnv, State
from rl.wrappers.baselines import JaxMARLWrapper, LogEnvState
from waymax import datatypes
from waymax import env as _env
from waymax import agents
from waymax import visualization

from waymax import config as _config
from waymax.datatypes.action import Action
from waymax.datatypes import observation
from waymax.datatypes import roadgraph
from waymax.datatypes.simulator_state import SimulatorState


@struct.dataclass
class WaymaxLogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class WaymaxLogWrapper(JaxMARLWrapper):
    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, scenario, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(scenario, key)
        state = WaymaxLogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WaymaxLogEnvState,
        action: Union[int, float],
        scenario,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, scenario
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        # new_won_episode = (batch_reward >= 1.0).astype(jnp.float32)
        state = WaymaxLogEnvState(
            env_state=env_state,
            # won_episode=new_won_episode * (1 - ep_done),
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info



OBSERVE_ROADGRAPH = False


class WaymaxWrapper(JaxMARLWrapper):
    """
    Provides a `"world_state"` observation for the centralised critic.
    world state observation of dimension: (num_agents, world_state_size)    
    """
    
    def __init__(self,
                 env: _env.MultiAgentEnvironment,
                 obs_with_agent_id=False):
        super().__init__(env)
        self.num_agents = self._env.config.max_num_objects
        self.obs_with_agent_id = obs_with_agent_id
        self.agents = [
            f'object_{i}' for i in range(self.num_agents)
        ]
        
        # Size of each agent's observation (including other agents and rg points)
        self._agent_obs_size = self.num_agents * 7 + env.config.observation.roadgraph_top_k * 7

        self.world_state_fn = self.ws_just_env_state

        if not self.obs_with_agent_id:
            self._world_state_size = self._agent_obs_size
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._agent_obs_size + self.num_agents
            self.world_state_fn = self.ws_with_agent_id

        self.observation_spaces = {
            i: Box(low=-1, high=1.0, shape=(self._world_state_size,)) for i in self.agents
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
        
        # Get global sim trajectory
        global_traj = datatypes.dynamic_index(
            env_state.sim_trajectory, env_state.timestep, axis=-1, keepdims=True
        )
        
        obs = {}
        # agent_ids = jnp.eye(env.num_agents)

        for idx in range(env.config.max_num_objects):

            # Get global roadgraph points
            global_rg = roadgraph.filter_topk_roadgraph_points(
                env_state.roadgraph_points,
                env_state.sim_trajectory.xy[idx, env_state.timestep],
                topk=env.config.observation.roadgraph_top_k,
            )
                        
            # if env.coordinate_frame == _config.CoordinateFrame.GLOBAL:
            #     exp_traj = global_traj
            #     exp_rg = global_rg
            # elif env.coordinate_frame == _config.CoordinateFrame.OBJECT:
            pose = observation.ObjectPose2D.from_center_and_yaw(
                xy=env_state.sim_trajectory.xy[idx, env_state.timestep],
                yaw=env_state.sim_trajectory.yaw[idx, env_state.timestep],
                valid=env_state.sim_trajectory.valid[idx, env_state.timestep],
            )
            
            sim_traj = observation.transform_trajectory(global_traj, pose)
            exp_rg = observation.transform_roadgraph_points(global_rg, pose)
            
            agent_obs = jnp.concatenate((
                sim_traj.xyz.reshape(-1), sim_traj.yaw.reshape(-1),
                sim_traj.vel_xy.reshape(-1), sim_traj.vel_yaw.reshape(-1),
                exp_rg.xyz.reshape(-1), exp_rg.dir_xyz.reshape(-1),
                exp_rg.types.reshape(-1), # agent_ids[idx],
            ))
        
            obs[f'object_{idx}'] = agent_obs

        # Shape is (2000, 2)
        #valid_roadgraph_points_xy = roadgraph_points_xy

        # world_state = jnp.concatenate(
        # (exp_traj.xy.reshape(-1), exp_traj.yaw.reshape(-1), exp_traj.vel_x.reshape(-1), exp_traj.vel_y.reshape(-1),
        #  exp_traj.vel_yaw.reshape(-1)), axis=0
        # )
        # agent_ids = jnp.eye(env.num_agents)
        # world_state_w_ids = jnp.concatenate((world_state, agent_ids), axis=-1)
        # world_state_w_ids = world_state_w_ids[valid]
        # obs = {
        #     agent: jnp.concatenate((world_state, agent_ids[i]), axis=0)
        #     for i, agent in enumerate(env.agents)
        # }
        obs.update(
            {
                "world_state": jnp.stack([obs[agent] for agent in env.agents])
            }
        )
                
        return obs
        

    def get_avail_actions(self, state: SimulatorState):
        traj = datatypes.dynamic_index(
                state.sim_trajectory, state.timestep, axis=-1, keepdims=True
        )
        valid = traj.valid
        return {
            # agent: jnp.ones((2,)) for i, agent in enumerate(self.agents) if valid[i]
            agent: jnp.ones((2,)) for i, agent in enumerate(self.agents)
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
        # traj = datatypes.dynamic_index(
        #         state.sim_trajectory, state.timestep, axis=-1, keepdims=True
        # )
        # valid = traj.valid

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
        info = {}
        obs = self.get_obs(env_state)
        done = {'__all__': env_state.is_done}

        # TODO: Is this right? Is there something attached to the env_state if a controlled agent has e.g. gone out of bounds?
        #   Or does the `valid` value associated with an agent potentially change over the course of an episode?
        done.update(
            {agent: True for agent, valid_i in zip(self.agents, valid)}
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
    
    def render(self, scenario, env, key: jax.random.PRNGKey, actor=None, init_steps=11):
        """Make a video of the policy acting in the environment."""
        
        if actor is None:
            # IDM actor/policy controlling both object 0 and 1.
            # TODO: use our policy 
            actor = agents.IDMRoutePolicy(
                # Controlled objects are those valid at t=0.
                is_controlled_func=lambda state: state.log_trajectory.valid[..., init_steps]
            )
        
        actors = [actor]

        jit_step = jax.jit(env.step)
        jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

        init_obs, init_state = env.reset(scenario, key)
        frames = []
        rng = jax.random.PRNGKey(0)

        def step_env(carry, _):
            rng, obs, state = carry

            traj = datatypes.dynamic_index(
                state.sim_trajectory, state.timestep, axis=-1, keepdims=True
            )
            hidden, pi = actor.network.apply(actor.params, hidden, obs)            

            # outputs = [
            #     jit_select_action({}, state, obs, None, rng)
            #     for jit_select_action in jit_select_action_list
            # ]
            action = agents.merge_actions(outputs)
            next_state = jit_step(state, action)
            rng, _ = jax.random.split(rng)

            return (rng, next_state), next_state
            
        frames = [visualization.plot_simulator_state(init_state.env_state, use_log_traj=False)]
        remaining_timesteps = init_state.env_state.remaining_timesteps
        states = jax.lax.scan(step_env, (rng, init_obs, init_state), None, length=remaining_timesteps, reverse=False, unroll=5)
        for i in range(remaining_timesteps):
            if i % 5 == 0:
                state = jax.tree.map_structure(lambda x: x[i], states)
                frames.append(visualization.plot_simulator_state(state.env_state, use_log_traj=False))

        return np.array(frames)
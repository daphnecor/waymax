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
from waymax import datatypes
from waymax import config as _config
from waymax.datatypes.action import Action
from waymax.datatypes import observation
from waymax.datatypes import roadgraph
from waymax.datatypes.simulator_state import SimulatorState
from waymax.datatypes import get_control_mask
from waymax.datatypes import operations


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
        
        # Take a step in the environment
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
        # Shape explained: RG = (x, y, z, type), TL = (state, x, y, lane_id)
        # 6 = (traj_info, yaw_info, vel_xy_info, roadgraph_info)
        self._agent_obs_size = ( 
            self.num_agents * 7 + env.config.observation.roadgraph_top_k * 4 #+ (4 * 16) # TLS
        )

        self.world_state_fn = self.ws_just_env_state

        if not self.obs_with_agent_id:
            self._world_state_size = self._agent_obs_size
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._agent_obs_size + self.num_agents
            self.world_state_fn = self.ws_with_agent_id

        self.observation_spaces = {
            i: Box(low=-1.0, high=1.0, shape=(self._world_state_size,)) for i in self.agents
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
        
        MAX_REL_XYZ = 25_000
        MIN_REL_XYZ = -25_000
        MAX_VEL_XYZ = 70
        MIN_VEL_XYZ = -70
        
        def _norm(x, min_val, max_val):
            """Normalize to range [-1, 1]."""
            return 2 * ((x - min_val) / (max_val - min_val)) - 1
   
        # Get global sim trajectory
        global_traj = datatypes.dynamic_index(
            env_state.sim_trajectory, env_state.timestep, axis=-1, keepdims=True
        )
        
        # Global traffic light information
        # Shape is (num_tls = 16, 1) --> see datatypes/traffic_lights.py
        current_global_tl = operations.dynamic_slice(
            env_state.log_traffic_light, jnp.array(env_state.timestep, int), 1, axis=-1
        )
        
        obs = {}

        for idx in range(env.config.max_num_objects):

            # Get global roadgraph points
            global_rg = roadgraph.filter_topk_roadgraph_points(
                env_state.roadgraph_points,
                env_state.sim_trajectory.xy[idx, env_state.timestep],
                topk=env.config.observation.roadgraph_top_k,
            )
            
            # Get agent pose: Position, orientation, and rotation matrix
            pose = observation.ObjectPose2D.from_center_and_yaw(
                xy=env_state.sim_trajectory.xy[idx, env_state.timestep],
                yaw=env_state.sim_trajectory.yaw[idx, env_state.timestep],
                valid=env_state.sim_trajectory.valid[idx, env_state.timestep],
            )
            
            # Transform to relative coordinates using agent i's pose
            sim_traj = observation.transform_trajectory(global_traj, pose)
            local_rg = observation.transform_roadgraph_points(global_rg, pose)
            local_tl = observation.transform_traffic_lights(current_global_tl, pose)
            
            # Unpack traffic lights, there are a maximum of 16 traffic lights per scene
            # Not all traffic lights are valid
            valid_tl_ids = local_tl.valid.reshape(-1)
            
            tl_valid_states = jnp.where(valid_tl_ids, local_tl.state.reshape(-1), 0)
            tl_x_valid = jnp.where(valid_tl_ids, local_tl.x.reshape(-1), 0)
            tl_y_valid = jnp.where(valid_tl_ids, local_tl.y.reshape(-1), 0)
            tl_lane_ids_valid = jnp.where(valid_tl_ids, local_tl.lane_ids.reshape(-1), 0)
                        
            traj_info = sim_traj.xyz.reshape(-1)
            yaw_info = sim_traj.yaw.reshape(-1)
            vel_xy_info = sim_traj.vel_xy.reshape(-1)
            vel_yaw_info = sim_traj.vel_yaw.reshape(-1)
            
            # Road graph information: x, y, z, type
            roadgraph_info = jnp.concatenate(
                (local_rg.xyz.reshape(-1), 
                local_rg.dir_xyz.reshape(-1),
                local_rg.types.reshape(-1)),
            )
            
            # # Construct agent observation
            # agent_obs = jnp.concatenate((
            #     traj_info, 
            #     yaw_info,
            #     vel_xy_info,
            #     vel_yaw_info,
            #     roadgraph_info,
            #     tl_valid_states,
            #     tl_x_valid,
            #     tl_y_valid,
            #     tl_lane_ids_valid
            # ))
            
            #traj_info = _norm(traj_info, MIN_REL_XYZ, MAX_REL_XYZ)
            #yaw_info = yaw_info / 2 * jnp.pi
            #vel_xy_info = _norm(vel_xy_info, MIN_VEL_XYZ, MAX_VEL_XYZ)
            #vel_yaw_info = vel_yaw_info / 2 * jnp.pi
            #roadgraph_info = _norm(roadgraph_info, MIN_REL_XYZ, MAX_REL_XYZ)
            tl_x_valid = _norm(tl_x_valid, MIN_REL_XYZ, MAX_REL_XYZ)
            tl_y_valid = _norm(tl_y_valid, MIN_REL_XYZ, MAX_REL_XYZ)
                    
            # Construct agent observation
            agent_obs = jnp.concatenate((
                traj_info, 
                yaw_info,
                vel_xy_info,
                roadgraph_info,
                #tl_valid_states,
                #tl_x_valid,
                #tl_y_valid,
                #tl_lane_ids_valid
            ))
                
            obs[f'object_{idx}'] = agent_obs
    
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
        
    def _get_control_mask(self, state: SimulatorState):
        control_mask = get_control_mask(state.object_metadata, _config.ObjectType.VALID)
        return control_mask
    
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
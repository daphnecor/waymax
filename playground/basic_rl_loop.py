import jax
from jax import numpy as jnp
import numpy as np
import mediapy
from tqdm import tqdm
import dataclasses
import wandb
from time import perf_counter

from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import agents
from waymax import visualization
from waymax.config import LinearCombinationRewardConfig

def render_frames(states):
    """Create wandb videos from simulator states.
    Returns:
        np.array: frames of shape (num_frames, channels, height, width)
    """
    frames = []
    for state in states:
        img = visualization.plot_simulator_state(state, use_log_traj=False)
        frames.append(img.T)
    return np.array(frames).astype(np.uint8)

if __name__ == "__main__":

    # Config dataset:
    max_num_objects = 128

    config = dataclasses.replace(_config.WOD_1_0_0_VALIDATION, max_num_objects=max_num_objects)
    
    dataset_config = dataclasses.replace(
        _config.DatasetConfig,
        batch_dims=(2,),
    )
    data_iter = dataloader.simulator_state_generator(config=config)
    scenarios = next(data_iter)
    
    jax.debug.breakpoint()

    # Config the multi-agent environment:
    init_steps = 11

    # Set the dynamics model the environment is using.
    # Note each actor interacting with the environment needs to provide action
    # compatible with this dynamics model.
    dynamics_model = dynamics.StateDynamics()
    
    # Change coordinates to relative
    obs_config = dataclasses.replace(
        _config.ObservationConfig(),
        coordinate_frame=_config.CoordinateFrame.OBJECT,
    )

    # Create env config
    env_config = dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=max_num_objects,
        observation=obs_config,
        rewards=LinearCombinationRewardConfig(
            {
                'overlap': -1.0, 
                'offroad': -1.0, 
                'log_divergence': 1.0
            }
        ),
        #metrics={}
        # Controll all valid objects in the scene.
        controlled_object=_config.ObjectType.VALID,
    )

    # Expect users to control all valid object in the scene.
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=env_config,
    )
    
    # Storage
    rewards = {}
    for agent_idx in range(max_num_objects):
        rewards[f"agent_{agent_idx}"] = []

    # Setup a few actors
    obj_idx = jnp.arange(max_num_objects)
    
    # # Rule-based agents
    # actor_0 = agents.IDMRoutePolicy(
    #     is_controlled_func=lambda state: (obj_idx == 0) | (obj_idx == 1)
    # )

    # Constant speed actor with predefined fixed speed controlling object 2.
    actor_1 = agents.create_constant_speed_actor(
        speed=5.0,
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: obj_idx == 2,
    )
    
    # Expert-controlled
    actor_2 = agents.create_expert_actor(
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: (obj_idx == 0) | (obj_idx == 1) | (obj_idx == 3) | (obj_idx == 4),
    )

    
    actors = [actor_1, actor_2]
    
    jit_step = jax.jit(env.step)
    jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]
    
    states = [env.reset(scenario)]
    rng = jax.random.PRNGKey(0)
    
    run = wandb.init(
        project="waymax", 
        group="basic_rl_loop"
    )
    
    for time_step in range(states[0].remaining_timesteps):
        
        print(f"Step: {time_step}")
        
        current_state = states[-1]

        outputs = [
            jit_select_action({}, current_state, None, rng)
            for jit_select_action in jit_select_action_list
        ]
        
        action = agents.merge_actions(outputs)
        reward = env.reward(current_state, action)
        
        next_state = jit_step(current_state, action)        
        
        rng, _ = jax.random.split(rng)

        # Store
        states.append(next_state)
        for agent_idx in range(max_num_objects):
            wandb.log({f'reward_agent_{agent_idx}': np.asarray(reward[agent_idx])})
        
        print(f"SHAPES | state: {current_state.shape} action: {action.shape} reward: {reward.shape} next_state: {next_state.shape} \n")
        print(f"REWARDS \n {reward} \n")
    
    
    # Visualize the scene
    frames = render_frames(states)
    
    
    wandb.log({"video": wandb.Video(frames, fps=1, format="gif")})
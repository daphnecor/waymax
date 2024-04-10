# # Multi-agent Simulation
# 
# This tutorial demonstrates how to run a simple closed-loop simulation with multiple pre-defined sim agents.



import jax
from jax import numpy as jnp
from rl.model import Dense
import imageio
import numpy as np
import mediapy
from tqdm import tqdm
import dataclasses

from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import agents
from waymax import visualization




# Config dataset:
max_num_objects = 32

# Set the dynamics model the environment is using.
# Note each actor interacting with the environment needs to provide action
# compatible with this dynamics model.
# dynamics_model = dynamics.StateDynamics()
dynamics_model = dynamics.InvertibleBicycleModel()

# Setup a few actors, see visualization below for how each actor behaves.

# An actor that doesn't move, controlling all objects with index > 4
obj_idx = jnp.arange(max_num_objects)
static_actor = agents.create_constant_speed_actor(
    speed=0.0,
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: obj_idx > 4,
)

# # # IDM actor/policy controlling both object 0 and 1.
# # # Note IDM policy is an actor hard-coded to use dynamics.StateDynamics().
# # actor_0 = agents.IDMRoutePolicy(
# #     is_controlled_func=lambda state: (obj_idx == 0) | (obj_idx == 1)
# # )

# # # Constant speed actor with predefined fixed speed controlling object 2.
# actor_1 = agents.create_constant_speed_actor(
#     speed=5.0,
#     dynamics_model=dynamics_model,
#     is_controlled_func=lambda state: (obj_idx == 1) | (obj_idx == 2),
# )

import rl
# Exper/log actor controlling objects 3 and 4.
# actor_2 = agents.create_expert_actor(
# # dum_actor = dam.agents.create_dummy_actor(
#     dynamics_model=dynamics_model,
#     is_controlled_func=lambda state: obj_idx == 3,
# )

rng = jax.random.PRNGKey(0)

network = Dense(action_dim=2)
init_x = jnp.zeros((1, 6))
network_params = network.init(rng, init_x)
print(network.tabulate(rng, init_x))

dum_actor = rl.agents.create_netty_actor(
    network=network,
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: obj_idx <= 4,
)

# actors = [static_actor, actor_1, actor_2, dum_actor]
actors = [static_actor, dum_actor]

config = dataclasses.replace(_config.WOD_1_0_0_VALIDATION, max_num_objects=max_num_objects)
data_iter = dataloader.simulator_state_generator(config=config)
scenario = next(data_iter)


# ## Initializing and Running the Simulator
# 
# Waymax uses a Gym-like interface for running closed-loop simulation. 
# 
# The `env.MultiAgentEnvironment` class defines a stateless simulation interface with the two key methods:
# - The `reset` method initializes and returns the first simulation state.
# - The `step` method transitions the simulation and takes as arguments a state and an action and outputs the next state.
# 
# Crucially, the `MultiAgentEnvironment` does not hold any simulation state itself, and the `reset` and `step` functions have no side effects. This allows us to use functional transforms from JAX, such as using jit compilation to optimize the compuation. It also allows the user to arbitrarily branch and restart simulation from any state, or save the simulation by simply serializing and saving the state object.
# 
# 



# Config the multi-agent environment:
init_steps = 11


# Expect users to control all valid object in the scene.
env = _env.MultiAgentEnvironment(
    dynamics_model=dynamics_model,
    config=dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=max_num_objects,
        controlled_object=_config.ObjectType.VALID,
    ),
)


jit_step = jax.jit(env.step)
jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

states = [env.reset(scenario)]
for _ in range(states[0].remaining_timesteps):
  current_state = states[-1]

  outputs = [
      jit_select_action(params=network_params, state=current_state, actor_state=None, rng=rng)
      for jit_select_action in jit_select_action_list
  ]
  action = agents.merge_actions(outputs)
  next_state = jit_step(current_state, action)
  rng, _ = jax.random.split(rng)

  states.append(next_state)


# ## Visualization of simulation.
# 
# We can now visualize the result of the simulation loop.
# 
# On the left side:
# - Objects 5, 6, and 7 (controlled by static_actor) remain static.
# - Objects 3 and 4 controlled by log playback, and collide with objects 5 and 6.
# 
# On the right side:
# - Object 2 controlled by actor_1 is moving at constant speed 5m/s (i.e. slower than log in this case).
# - Object 0 and 1, controlled by the IDM agent, follow the log in the beginning, but object 1 slows down when approaching object 2.



imgs = []
states = jax.device_put(states, jax.devices('cpu')[0])
for state in states:
  imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))
# Save the video using  
imageio.v3.imwrite('demo.gif', imgs, duration=10)
# imageio.v3.imwrite('demo.mp4', imgs, fps=20)

# mediapy.show_video(imgs, fps=10)


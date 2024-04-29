import jax
from jax import numpy as jnp
import numpy as np
import wandb
import dataclasses

from waymax import config as _config
from waymax import dataloader
from waymax import dynamics
from waymax import env as _env
from waymax import agents
from waymax import visualization
from time import perf_counter

# Configuration
max_num_objects = 32

if __name__ == "__main__":

    # Render 
    run = wandb.init(
        project="waymax",
        group="render"
    )

    # Create the data iterator
    config = dataclasses.replace(_config.WOD_1_0_0_VALIDATION, max_num_objects=max_num_objects)
    data_iter = dataloader.simulator_state_generator(config=config)

    # Get a scene
    scenario = next(data_iter)

    # Config the multi-agent environment:
    init_steps = 11

    # Set the dynamics model the environment is using.
    # Note each actor interacting with the environment needs to provide action
    # compatible with this dynamics model.
    dynamics_model = dynamics.StateDynamics()

    make_env_start = perf_counter()

    # Expect users to control all valid object in the scene.
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(
            _config.EnvironmentConfig(),
            max_num_objects=max_num_objects,
            controlled_object=_config.ObjectType.VALID,
        ),
    )

    # Setup a few actors, see visualization below for how each actor behaves.
    # An actor that doesn't move, controlling all objects with index > 4
    obj_idx = jnp.arange(max_num_objects)
    static_actor = agents.create_constant_speed_actor(
        speed=0.0,
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: obj_idx > 4,
    )

    # IDM actor/policy controlling both object 0 and 1.
    # Note IDM policy is an actor hard-coded to use dynamics.StateDynamics().
    actor_0 = agents.IDMRoutePolicy(
        is_controlled_func=lambda state: (obj_idx == 0) | (obj_idx == 1)
    )

    # Constant speed actor with predefined fixed speed controlling object 2.
    actor_1 = agents.create_constant_speed_actor(
        speed=5.0,
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: obj_idx == 2,
    )

    # Exper/log actor controlling objects 3 and 4.
    actor_2 = agents.create_expert_actor(
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: (obj_idx == 3) | (obj_idx == 4),
    )

    actors = [static_actor, actor_0, actor_1, actor_2]

    jit_step = jax.jit(env.step)
    jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

    # Run simulation
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

    # Render the states!
    for state in states:
        img = visualization.plot_simulator_state(state, use_log_traj=False, highlight_obj=_config.ObjectType.VALID,)
        frames.append(img.T)
    end_sim = perf_counter()

    wandb.log({"video": wandb.Video(np.array(frames).astype(np.uint8), fps=10)})

    run.finish()

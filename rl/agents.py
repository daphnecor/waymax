from functools import partial
from typing import Callable, Optional

from flax import linen as nn
import jax
import jax.numpy as jnp

from waymax import datatypes
from waymax import dynamics
from waymax.agents import actor_core
from waymax.agents import waypoint_following_agent


def create_netty_actor(
    network: nn.Module,
    params,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
) -> actor_core.WaymaxActorCore:
  """Creates an actor with constant speed without changing objects' heading.

  Note the difference against ConstantSpeedPolicy is that an actor requires
  input of a dynamics model, while a policy does not (it assumes to use
  StateDynamics).

  Args:
    dynamics_model: The dynamics model the actor is using that defines the
      action output by the actor.
    is_controlled_func: Defines which objects are controlled by this actor.

  Returns:
    An stateless actor that drives the controlled objects with constant speed.
  """

  # TODO: Initialize the environment here

  def select_action(  # pytype: disable=annotation-type-mismatch
      params: actor_core.Params,
      state: datatypes.SimulatorState,
      actor_state=None,
      rng: jax.Array = None,
      network: nn.Module = None,
  ) -> actor_core.WaymaxActorOutput:
    """Computes the actions using the given dynamics model and speed."""

    del actor_state

    is_controlled = is_controlled_func(state)
    traj = datatypes.dynamic_index(
        state.sim_trajectory, state.timestep, axis=-1, keepdims=True
    )

    obs = env.get_obs(state)

    # obs = jnp.concatenate(
    #  (traj.xy, traj.yaw[..., None], traj.vel_x[..., None], traj.vel_y[..., None], traj.vel_yaw[..., None]), axis=-1
    # )[:, 0]

    hidden, pi = network.apply(params, obs)

    actions = datatypes.Action(data=pi, valid=traj.valid)

    # Note here actions' valid could be different from is_controlled, it happens
    # when that object does not have valid trajectory from the previous
    # timestep.
    return actor_core.WaymaxActorOutput(
        actor_state=None,
        action=actions,
        is_controlled=is_controlled,
    )

  _select_action = partial(select_action, network=network)

  return actor_core.actor_core_factory(
      init=lambda rng, init_state: None,
      select_action=_select_action,
      name=f'netty_actor',
  )



def create_randy_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
    speed: Optional[float] = None,
) -> actor_core.WaymaxActorCore:
  """Creates an actor with constant speed without changing objects' heading.

  Note the difference against ConstantSpeedPolicy is that an actor requires
  input of a dynamics model, while a policy does not (it assumes to use
  StateDynamics).

  Args:
    dynamics_model: The dynamics model the actor is using that defines the
      action output by the actor.
    is_controlled_func: Defines which objects are controlled by this actor.
    speed: Speed of the actor, if None, speed from previous step is used.

  Returns:
    An statelss actor that drives the controlled objects with constant speed.
  """

  def select_action(  # pytype: disable=annotation-type-mismatch
      params: actor_core.Params,
      state: datatypes.SimulatorState,
      actor_state=None,
      rng: jax.Array = None,
  ) -> actor_core.WaymaxActorOutput:
    """Computes the actions using the given dynamics model and speed."""
    del params, actor_state  # unused.
    traj_t0 = datatypes.dynamic_index(
        state.sim_trajectory, state.timestep, axis=-1, keepdims=True
    )
    if speed is None:
      vel_x = traj_t0.vel_x
      vel_y = traj_t0.vel_y
    else:
      vel_x = speed * jnp.cos(traj_t0.yaw)
      vel_y = speed * jnp.sin(traj_t0.yaw)

    is_controlled = is_controlled_func(state)
    traj_t1 = traj_t0.replace(
        x=traj_t0.x + vel_x * datatypes.TIME_INTERVAL,
        y=traj_t0.y + vel_y * datatypes.TIME_INTERVAL,
        vel_x=vel_x,
        vel_y=vel_y,
        valid=is_controlled[..., jnp.newaxis] & traj_t0.valid,
        timestamp_micros=(
            traj_t0.timestamp_micros + datatypes.TIMESTEP_MICROS_INTERVAL
        ),
    )

    traj_combined = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate((x, y), axis=-1), traj_t0, traj_t1
    )
    actions = dynamics_model.inverse(
        traj_combined, state.object_metadata, timestep=0
    )
    # jax.debug.breakpoint()
    rand_data = jax.random.uniform(rng, actions.data.shape) * 100
    # rand_data = rand_data.at[:, 0].set(rand_data[:, 0] * 6 - 3)
    # rand_data = rand_data.at[:, 1].set(rand_data[:, 0] * 1.2 - 0.6)
    # breakpoint()
    # jax.debug.breakpoint()

    actions = actions.replace(data=rand_data)

    # Note here actions' valid could be different from is_controlled, it happens
    # when that object does not have valid trajectory from the previous
    # timestep.
    return actor_core.WaymaxActorOutput(
        actor_state=None,
        action=actions,
        is_controlled=is_controlled,
    )

  return actor_core.actor_core_factory(
      init=lambda rng, init_state: None,
      select_action=select_action,
      name=f'constant_speed_{speed}',
  )

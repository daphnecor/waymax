import os

import hydra
import imageio
import jax
from jax import numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from orbax import checkpoint as ocp
from waymax import datatypes
from waymax import visualization

from rl.config.config import EnjoyConfig
from rl.mappo import RunnerState, init_config
from rl.model import ActorRNN
from rl.wrappers.marl import WaymaxWrapper
from utils import batchify, init_run, restore_run, unbatchify


@hydra.main(version_base=None, config_path="config", config_name="enjoy")
def enjoy(config: EnjoyConfig):
    init_config(config)

    # TODO: Render multiple episodes, generate states efficiently by vmapping.
    config.NUM_ENVS = 1

    rng = jax.random.PRNGKey(config.SEED)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        config.ckpt_dir, options=options)
    runner_state: RunnerState
    env: WaymaxWrapper
    latest_update_step = checkpoint_manager.latest_step()
    runner_state, env, scenario, latest_update_step = \
        init_run(config, checkpoint_manager, latest_update_step, rng)

    if not config.random_agent:
        runner_state, _ = restore_run(config, runner_state, checkpoint_manager, latest_update_step)

    actor_network = ActorRNN(env.action_space(env.agents[0]).shape[0], config=config)

    rng = jax.random.PRNGKey(0)
    init_obs, init_state = env.reset(scenario, rng)
    actor_params = runner_state.train_states[0].params
    actor_hidden = runner_state.hstates[0]

    def step_env(carry, _):
        rng, obs, state, done, actor_hidden = carry
        # print(obs.shape)

        # traj = datatypes.dynamic_index(
        #     state.env_state.sim_trajectory, state.env_state.timestep, axis=-1, keepdims=True
        # )

        if not config.random_agent:
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
                action, env.agents, config.NUM_ENVS, env.num_agents
            )
            env_act = {k: v.squeeze() for k, v in env_act.items()}
        else:
            rng_acts = jax.random.split(rng, len(env.agents))
            agent_acts = jax.vmap(env.action_space(env.agents[0]).sample, in_axes=(0,))(rng_acts)
            env_act = {agent: act for agent, act in zip(env.agents, agent_acts)}

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

    remaining_timesteps = init_state.env_state.remaining_timesteps
        
    done = jnp.zeros((len(env.agents),), dtype=bool)
    init_obs = batchify(init_obs, env.agents, env.num_agents)

    _, states = jax.lax.scan(step_env, (rng, init_obs, init_state, done, actor_hidden), None, length=remaining_timesteps)

    # states = jax.tree.map(lambda x, y: jnp.concatenate([x[None], y], axis=0), init_state, states)

    # states = []
    # rng, obs, state, done, actor_hidden = (rng, init_obs, init_state, done, actor_hidden)
    # for i in range(remaining_timesteps):
    #     carry, state = step_env((rng, obs, state, done, actor_hidden), None)
    #     rng, obs, state, done, actor_hidden = carry
    #     states.append(state)

    # frames = [visualization.plot_simulator_state(init_state.env_state, use_log_traj=False)]
    frames = []
    for i in range(remaining_timesteps):
        if i % 1 == 0:
            # state = jax.tree.map(lambda x: x[i] if len(x.shape) > 0 else x, states)
            state = jax.tree.map(lambda x: x[i], states)
            # Print all leaf names
            frames.append(visualization.plot_simulator_state(state.env_state, use_log_traj=False))
    
    os.makedirs(config.vid_dir, exist_ok=True)

    if config.random_agent:
        vid_path = os.path.join(config.vid_dir, f"enjoy_random_agent.gif")
    else:
        vid_path = os.path.join(config.vid_dir, f"enjoy_{latest_update_step}.gif")
    imageio.mimsave(vid_path, frames, fps=10, loop=0)

    
if __name__ == '__main__':
    enjoy()



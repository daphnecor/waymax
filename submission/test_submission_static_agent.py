
import os
import jax
from jax import random
from jax import numpy as jnp
import tensorflow as tf

from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymax import agents
from waymax import config as _config
from waymax import dynamics
from waymax import dataloader
from waymax import datatypes
from waymax import env as _env

CURRENT_TIME_INDEX = 10
N_SIMULATION_STEPS = 80
N_ROLLOUTS = 32

if __name__ == "__main__":
    
    # Specify which dataset to use
    data_config = _config.WOD_1_1_0_TRAINING
    
    # Create dataloader
    # Write a custom dataloader that loads scenario IDs.
    def _preprocess(serialized: bytes):
        womd_features = dataloader.womd_utils.get_features_description(
            include_sdc_paths=data_config.include_sdc_paths,
            max_num_rg_points=data_config.max_num_rg_points,
            num_paths=data_config.num_paths,
            num_points_per_path=data_config.num_points_per_path,
        )
        womd_features['scenario/id'] = tf.io.FixedLenFeature([1], tf.string)

        deserialized = tf.io.parse_example(serialized, womd_features)
        parsed_id = deserialized.pop('scenario/id')
        deserialized['scenario/id'] = tf.io.decode_raw(parsed_id, tf.uint8)

        return dataloader.preprocess_womd_example(
            deserialized,
            aggregate_timesteps=data_config.aggregate_timesteps,
            max_num_objects=data_config.max_num_objects,
        )

    def _postprocess(example):
        scenario = dataloader.simulator_state_from_womd_dict(example)
        scenario_id = example['scenario/id']
        return scenario_id, scenario

    def decode_bytes(data_iter):
        for scenario_id, scenario in data_iter:
            scenario_id = scenario_id.tobytes().decode('utf-8')
            yield scenario_id, scenario

    data_iter = decode_bytes(dataloader.get_data_generator(
        data_config, _preprocess, _postprocess
    ))
    
    
    # Configure environment and agents 
    env_config = _config.EnvironmentConfig(
        # Ensure that the sim agent can control all valid objects.
        controlled_object=_config.ObjectType.VALID
    )

    dynamics_model = dynamics.InvertibleBicycleModel()
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=env_config,
    )

    agent = agents.create_constant_speed_actor(
        dynamics_model=dynamics_model,
        # Controlled objects are those valid at t=0.
        is_controlled_func=lambda state: state.log_trajectory.valid[..., CURRENT_TIME_INDEX]
    )

    # Take steps in the environment
    jit_step = jax.jit(env.step)
    jit_select_action = jax.jit(agent.select_action)
    
    # Generate rollouts
    # Define a function that will rollout the environment and agent to generate trajectories
    def validate_scenario_rollout(
        scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts, 
        scenario: datatypes.SimulatorState
    ):
        """Verifies if scenario_rollouts has correct formatting."""
        valid_sim_agents = scenario.log_trajectory.valid[..., CURRENT_TIME_INDEX]
        sim_agent_id_idxs = jnp.where(valid_sim_agents)[0]
        sim_agent_ids = scenario.object_metadata.ids[sim_agent_id_idxs].tolist()

        if len(scenario_rollouts.joint_scenes) != N_ROLLOUTS:
            raise ValueError('Incorrect number of parallel simulations. '
                            f'(Actual: {len(scenario_rollouts.joint_scenes)}, '
                            f'Expected: {N_ROLLOUTS})')

        def _raise_if_wrong_length(trajectory, field_name, expected_length):
            if len(getattr(trajectory, field_name)) != expected_length:
                raise ValueError(f'Invalid {field_name} tensor length '
                                f'(actual: {len(getattr(trajectory, field_name))}, '
                                f'expected: {expected_length})')

        for joint_scene in scenario_rollouts.joint_scenes:
            simulated_ids = []
            for simulated_trajectory in joint_scene.simulated_trajectories:
                # Check the length of each of the simulated fields.
                _raise_if_wrong_length(simulated_trajectory, 'center_x', N_SIMULATION_STEPS)
                _raise_if_wrong_length(simulated_trajectory, 'center_y', N_SIMULATION_STEPS)
                _raise_if_wrong_length(simulated_trajectory, 'center_z', N_SIMULATION_STEPS)
                _raise_if_wrong_length(simulated_trajectory, 'heading', N_SIMULATION_STEPS)
                
                # Check that each object ID is present in the original WOMD scenario.
                if simulated_trajectory.object_id not in sim_agent_ids:
                    raise ValueError(
                        f'Object {simulated_trajectory.object_id} is not a sim agent.')
                simulated_ids.append(simulated_trajectory.object_id)
            
            # Check that all of the required objects/agents are simulated.
            missing_agents = set(sim_agent_ids) - set(simulated_ids)
            if missing_agents:
                raise ValueError(
                    f'Sim agents {missing_agents} are missing from the simulation.')

    def generate_scenario_rollout(
        scenario_id: str,
        scenario: datatypes.SimulatorState) -> sim_agents_submission_pb2.ScenarioRollouts:
        """Simulate 32 rollouts and return a ScenarioRollouts protobuf."""
        joint_scenes = []
        key = random.PRNGKey(0)
        for _ in range(N_ROLLOUTS):
            initial_state = current_state = env.reset(scenario)
            # Controlled objects are those valid at t=0.
            is_controlled = scenario.log_trajectory.valid[..., CURRENT_TIME_INDEX]

            # Run the sim agent for 80 steps.
            for _ in (range(initial_state.remaining_timesteps)):
                key, actor_key = random.split(key, 2)
                actor_output = jit_select_action({}, current_state, None, actor_key)
                next_state = jit_step(current_state, actor_output.action)
                current_state = next_state

            # Write out result
            final_trajectory = current_state.sim_trajectory
            object_ids = current_state.object_metadata.ids  # Shape (n_objects,)
            object_ids = jnp.where(is_controlled, object_ids, -1)

            simulated_trajectories = []
            for i, object_id in enumerate(object_ids):
                if object_id != -1:
                    simulated_trajectory = sim_agents_submission_pb2.SimulatedTrajectory(
                            center_x=final_trajectory.x[i, env_config.init_steps:],
                            center_y=final_trajectory.y[i, env_config.init_steps:],
                            center_z=final_trajectory.z[i, env_config.init_steps:],
                            heading=final_trajectory.yaw[i, env_config.init_steps:],
                            object_id=object_id,
                    )
                    simulated_trajectories.append(simulated_trajectory)
            joint_scene = sim_agents_submission_pb2.JointScene(
                    simulated_trajectories=simulated_trajectories
            )
            joint_scenes.append(joint_scene)

        scenario_rollouts =  sim_agents_submission_pb2.ScenarioRollouts(
            scenario_id=scenario_id, joint_scenes=joint_scenes
        )
        validate_scenario_rollout(scenario_rollouts, scenario)
        return scenario_rollouts
        
    # Generate submission
    OUTPUT_ROOT_DIRECTORY = '/tmp/waymo_sim_agents/'
    os.makedirs(OUTPUT_ROOT_DIRECTORY, exist_ok=True)
    output_filenames = []
    scenario_rollouts = []

    for i, (scenario_id, scenario) in enumerate(data_iter):
        scenario_rollouts.append(generate_scenario_rollout(scenario_id, scenario))

        if i % 5 == 0 and i > 0:
            shard_suffix = '.%d' % i
            shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
                scenario_rollouts=scenario_rollouts,
                submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
                account_name='your_account@test.com',
                unique_method_name='waymax_sim_agents_tutorial',
                authors=['test'],
                affiliation='waymo',
                description='Submission from the Waymax - Sim Agents tutorial',
                method_link='https://waymo.com/open/'
            )
            scenario_rollouts = []
            output_filename = f'submission.binproto{shard_suffix}'
            with open(os.path.join(OUTPUT_ROOT_DIRECTORY, output_filename), 'wb') as f:
                f.write(shard_submission.SerializeToString())
            output_filenames.append(output_filename)

    # Once we have created all the shards, we can package them directly into a
    # tar.gz archive, ready for submission.
    with tarfile.open(
        os.path.join(OUTPUT_ROOT_DIRECTORY, 'submission.tar.gz'), 'w:gz') as tar:
        for output_filename in output_filenames:
            tar.add(os.path.join(OUTPUT_ROOT_DIRECTORY, output_filename),
                    arcname=output_filename)

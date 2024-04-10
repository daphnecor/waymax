from waymax import env, config, dynamics, datatypes
from waymax import dataloader


# Initialization
dynamics_model = dynamics.InvertibleBicycleModel()
env_config = config.EnvironmentConfig()
scenarios = dataloader.simulator_state_generator(config.WOD_1_1_0_TRAINING)
waymax_env = env.MultiAgentEnvironment(dynamics_model, env_config)

# Rollout
state = waymax_env.reset(next(scenarios))
total_returns = 0
while not state.is_done:
    action_spec = waymax_env.action_spec()
    action = datatypes.Action(data=action_spec.data.generate_value(), valid=action_spec.valid.generate_value())
    total_returns += waymax_env.reward(state, action)
    state = waymax_env.step(state, action)
    print(state)
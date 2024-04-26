import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import dataclasses
from torch.distributions import Normal
import jax

from waymax import config as _config
from waymax import dataloader
from waymax import visualization
from waymax import env as _env
import jax.numpy as jnp
from waymax import dynamics

def extract_features_and_labels(log_trajectory):
    num_objects = log_trajectory.num_objects
    num_timesteps = log_trajectory.num_timesteps

    evaluated_xyz = jax.device_get(log_trajectory.xyz[:, :-1, :])
    evaluated_vel = jax.device_get(log_trajectory.vel_xy[:, :-1, :])
    evaluated_yaw = jax.device_get(log_trajectory.yaw[:, :-1])

    # Ensure all tensors are 3D
    evaluated_xyz = torch.tensor(evaluated_xyz, dtype=torch.float32)
    evaluated_vel = torch.tensor(evaluated_vel, dtype=torch.float32)
    evaluated_yaw = torch.tensor(evaluated_yaw, dtype=torch.float32).unsqueeze(-1)  # Adding dimension

    features = torch.cat([
        evaluated_xyz,
        evaluated_vel,
        evaluated_yaw,
    ], dim=-1)

    next_xyz = jax.device_get(log_trajectory.xyz[:, 1:, :])
    next_vel = jax.device_get(log_trajectory.vel_xy[:, :-1, :])
    next_yaw = jax.device_get(log_trajectory.yaw[:, :-1])
    labels = torch.cat([
        torch.tensor(next_xyz, dtype=torch.float32),
        torch.tensor(next_vel, dtype=torch.float32),
        torch.tensor(next_yaw, dtype=torch.float32).unsqueeze(-1),  # Ensure same dimensions
    ], dim=-1)

    features = features.view(-1, features.shape[-1])
    labels = labels.view(-1, labels.shape[-1])

    return features, labels

def post_process_actions(raw_output, num_objects, num_features_per_object):

    reshaped_output = raw_output.view(num_objects, num_features_per_object)
    return reshaped_output


dynamics_model = dynamics.StateDynamics()

max_num_objects = 32
env = _env.MultiAgentEnvironment(
    dynamics_model=dynamics_model,
    config=dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=max_num_objects,
        controlled_object=_config.ObjectType.VALID,
    ),
)

class BCNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BCNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.tanh2 = nn.Tanh()
        self.mean = nn.Linear(hidden_size, output_size)
        self.log_std = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mean, std

def train_model(data_iter, model, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for _ in range(num_batches_per_epoch):
            try:
                scenario = next(data_iter)
                states, actions = extract_features_and_labels(scenario.log_trajectory)
                states, actions = states.to(device), actions.to(device)

                optimizer.zero_grad()
                mean, std = model(states)
                dist = Normal(mean, std)
                loss = -dist.log_prob(actions).sum()
                loss.backward()
                optimizer.step()
            except StopIteration:
                break
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        
        
def test_model(env, model, device, scenario):
    model.eval()
    # Reset the environment to start the test and extract the initial state and labels for features
    initial_state = env.reset(scenario)
    state_features, _ = extract_features_and_labels(initial_state.log_trajectory)
    state_tensor = torch.tensor(state_features, dtype=torch.float32, device=device)
    
    done = False
    while not done:
        with torch.no_grad():
            mean, _ = model(state_tensor.unsqueeze(0))  
            action = mean.squeeze(0).cpu().numpy()  
        reshaped_actions = post_process_actions(action, max_num_objects, 6)
        
        valid_mask = jax.numpy.ones((num_objects, 1), dtype=bool) 
        actions = Action(data=jax.numpy.array(reshaped_actions), valid=valid_mask)

        next_state, done = env.step(initial_state, actions)
        visualization.plot_simulator_state(next_state)

        state_features, _ = extract_features_and_labels(next_state.log_trajectory)
        state_tensor = torch.tensor(state_features, dtype=torch.float32, device=device)
        
        initial_state = next_state

        


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _config.WOD_1_0_0_VALIDATION
    env_config = _config.EnvironmentConfig(controlled_object=_config.ObjectType.VALID)

    data_iter = dataloader.simulator_state_generator(dataclasses.replace(_config.WOD_1_1_0_TRAINING, max_num_objects=32))

    model = BCNetwork(input_size=6, hidden_size=128, output_size=6)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

#number of batches?
    num_batches_per_epoch = 100
    train_model(data_iter, model, optimizer, device, epochs=50)

    torch.save(model.state_dict(), 'bc_model.pth')
    

    scenario = next(data_iter)  
    test_model(env, model, device, scenario)

    



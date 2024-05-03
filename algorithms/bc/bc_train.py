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
from tqdm import tqdm
import numpy as np
import mediapy




def extract_features_and_labels_for_timestep(log_trajectory, t):
    features = torch.cat([
        torch.tensor(jax.device_get(log_trajectory.xyz[:, t:t+1, :]), dtype=torch.float32),
        torch.tensor(jax.device_get(log_trajectory.yaw[:, t:t+1]), dtype=torch.float32).unsqueeze(-1),
        torch.tensor(jax.device_get(log_trajectory.vel_xy[:, t:t+1, :]), dtype=torch.float32),
    ], dim=-1)

    labels = torch.cat([
        torch.tensor(jax.device_get(log_trajectory.xyz[:, t+1:t+2, :]), dtype=torch.float32),
        torch.tensor(jax.device_get(log_trajectory.yaw[:, t+1:t+2]), dtype=torch.float32).unsqueeze(-1),
        torch.tensor(jax.device_get(log_trajectory.vel_xy[:, t+1:t+2, :]), dtype=torch.float32),
    ], dim=-1)

    return features.view(-1, features.shape[-1]), labels.view(-1, labels.shape[-1])





dynamics_model = dynamics.StateDynamics()
max_num_objects = 32
env = _env.MultiAgentEnvironment(
    dynamics_model=dynamics.StateDynamics(),
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
    
  

    

def train_model(data_iter, model, optimizer, device, epochs, num_batches_per_epoch):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_mae = 0
        count = 0

        for _ in range(num_batches_per_epoch):
            try:
                scenario = next(data_iter)
                num_timesteps = scenario.log_trajectory.num_timesteps - 1

                for t in range(num_timesteps):
                    features, labels = extract_features_and_labels_for_timestep(scenario.log_trajectory, t)
                    features, labels = features.to(device), labels.to(device)

                    optimizer.zero_grad()
                    mean, std = model(features)
                    dist = Normal(mean, std)
                    loss = -dist.log_prob(labels).sum()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    mae = torch.mean(torch.abs(mean - labels)).item()
                    total_mae += mae
                    count += 1

            except StopIteration:
                data_iter = dataloader.simulator_state_generator(dataclasses.replace(_config.WOD_1_1_0_TRAINING, max_num_objects=32))
                break  
                

        average_loss = total_loss / count if count != 0 else 0
        average_mae = total_mae / count if count != 0 else 0
        print(f'Epoch {epoch+1}, Average Loss: {average_loss:.4f}, Average MAE: {average_mae:.4f}')

    # Ensure that the data iterator is reinitialized for the next epoch
    data_iter = dataloader.simulator_state_generator(dataclasses.replace(_config.WOD_1_1_0_TRAINING, max_num_objects=32))


  

   
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _config.WOD_1_1_0_TRAINING
    #config = _config.WOD_1_0_0_VALIDATION
    env_config = _config.EnvironmentConfig(controlled_object=_config.ObjectType.VALID)

    data_iter = dataloader.simulator_state_generator(dataclasses.replace(_config.WOD_1_1_0_TRAINING, max_num_objects=32))

    model = BCNetwork(input_size=6, hidden_size=128, output_size=6)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    
    train_model(data_iter, model, optimizer, device, epochs=20, num_batches_per_epoch = 300)

    torch.save(model.state_dict(), 'bc_model.pth')
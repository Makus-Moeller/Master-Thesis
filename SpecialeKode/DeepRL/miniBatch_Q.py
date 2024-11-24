import sys
import numpy as np
sys.path.insert(0, 'C:/Udvikler/Speciale/SpecialeKode')
from model_evaluation.eval_utilities import plot_loss_progress
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworkSkipConnection(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, bias=True):
        super(QNetworkSkipConnection, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size, bias)  
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias)  
        self.output_layer = nn.Linear(hidden_size + state_size, action_size, bias)

    def forward(self, x_input):
        x = F.tanh(self.fc1(x_input))
        x = F.tanh(self.fc2(x))
        x = torch.cat((x_input, x), dim=1)
        x = self.output_layer(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, bias=True):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size, bias)  
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias)  
        self.output_layer = nn.Linear(hidden_size, action_size, bias)

    def forward(self, x_input):
        x = F.relu(self.fc1(x_input))  # Using ReLU activation
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)  # Directly outputting Q-values
        return x

class MBQ:
    def __init__(self,
                 state_size,
                 action_size,
                 hidden_size,
                 seed=1
        ):
        self.state_size = state_size
        self.action_size = action_size
        self.set_random_seed(seed)
        self.q_net = QNetworkSkipConnection(state_size, action_size, hidden_size)

    def sample_batch(self, data, batch_size):
        '''data should be a list of SARS tuples'''
        idx = np.random.choice(len(data), batch_size, replace=False)
        return [data[i] for i in idx]

    def set_random_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def train(self, experience_data, train_iterations=1000, batch_size=128, lr=0.001, gamma=0.99, optimizer=None, loss_fn=None, seed=42):
        print(self.q_net)
        q_losses = []

        # Initialize optimizer and loss function   
        optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr) if optimizer == None else optimizer
        loss_fn = torch.nn.MSELoss() if loss_fn == None else loss_fn

        for iteration in range(train_iterations):  # Set the number of iterations or some stopping criterion

            # Sample a mini-batch from the fixed dataset (memory buffer)
            batch = self.sample_batch(experience_data, batch_size)  # Memory is assumed to be pre-filled with experience tuples
            states = np.array([each[0] for each in batch], dtype=np.float32)
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch], dtype=np.float32)
            next_states = np.array([each[3] for each in batch], dtype=np.float32)

            # Convert to tensors
            next_states_tensor = torch.as_tensor(next_states)  # No need to copy the data
            rewards_tensor = torch.as_tensor(rewards)
            states_tensor = torch.as_tensor(states)
            actions_tensor = torch.as_tensor(actions, dtype=torch.int64)
            
            # Compute Q values for the next state (without gradient)
            with torch.no_grad():
                target_Qs_tensor = self.q_net(next_states_tensor)
                
                # Set target Qs to 0 for terminal states (episode ends)
                episode_ends = (next_states == -1).all(axis=1)  # Identifying terminal states
                target_Qs_tensor[episode_ends] = torch.zeros(self.action_size)
                
                # Compute the target: r + gamma * max(Q(s', a'))
                targets_tensor = rewards_tensor + gamma * torch.max(target_Qs_tensor, dim=1)[0]

            # Compute the Q values for the actions taken
            output_tensor = self.q_net(states_tensor)  # Get Q-values for all actions in current state
            Q_tensor = torch.gather(output_tensor, 1, actions_tensor.unsqueeze(-1)).squeeze()  # Get Q-values for selected actions

            # Compute loss between predicted Q-values and target Q-values
            loss = loss_fn(Q_tensor, targets_tensor)
            q_losses.append(loss.item())

            # Gradient-based update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optionally, log or print some statistics to monitor progress
            if iteration % 100 == 0:
                print(f'Iteration: {iteration}, Loss: {loss.item():.4f}')
        plot_loss_progress(q_losses, labels=["Q Loss"])
            

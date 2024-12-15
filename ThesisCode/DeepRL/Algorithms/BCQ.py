import sys
import torch
import numpy as np
sys.path.insert(0, 'C:/Udvikler/Speciale/SpecialeKode')
from utils import *
from model_evaluation.eval_utilities import plot_loss_progress
import torch.nn as nn


# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        if action.dim() == 1: 
            action = action.unsqueeze(1) 
        
        sa = torch.cat([state, action], dim=1) 
        return self.network(sa)

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

# Define the Perturbation Network
class PerturbationNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PerturbationNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # The perturbation should be small
        )

    def forward(self, state, action):
        if action.dim() == 1:  
            action = action.unsqueeze(1)  
        
        sa = torch.cat([state, action], dim=1)
        return 0.1 * self.network(sa)  


class BCQ:
    def __init__(self, state_size, action_size, hidden_size, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.set_random_seed(seed)
        self.q_net = QNetwork(state_size, action_size, hidden_size)
        self.policy_net = PolicyNetwork(state_size, action_size, hidden_size)
        self.perturb_net = PerturbationNetwork(state_size, action_size, hidden_size)
        
    def sample_batch(self, experience_data, batch_size):
        '''data should be a list of SARS tuples'''
        idx = np.random.choice(len(experience_data), batch_size, replace=False)
        return [experience_data[i] for i in idx]

    def set_random_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(self, experience_data, train_iterations=1000, batch_size=128, lr=0.001, gamma=0.99, seed=42):
        print('QNet: ', self.q_net)
        print('policyNet: ', self.policy_net)
        print('PertubNet: ', self.perturb_net)
        q_losses = []
        policy_losses = []
        perturb_losses = []
        
        # Set optimizers
        q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        perturb_optimizer = torch.optim.Adam(self.perturb_net.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        for iteration in range(train_iterations):
            batch = self.sample_batch(experience_data, batch_size)
            states = np.array([each[0] for each in batch], dtype=np.float32)
            actions = np.array([each[1] for each in batch], dtype=np.float32)
            rewards = np.array([each[2] for each in batch], dtype=np.float32)
            next_states = np.array([each[3] for each in batch], dtype=np.float32)

            # Convert to tensors
            next_states_tensor = torch.as_tensor(next_states)
            rewards_tensor = torch.as_tensor(rewards)
            states_tensor = torch.as_tensor(states)
            actions_tensor = torch.as_tensor(actions)


            # --- Q-Network Update ---
            with torch.no_grad():
                next_actions = self.policy_net(next_states_tensor)
                perturbed_next_actions = next_actions + self.perturb_net(next_states_tensor, next_actions)

                next_q_values = self.q_net(next_states_tensor, perturbed_next_actions).squeeze(1)
                
                # Terminal state handling (assume terminal states are represented by all -1s in next_state)
                episode_ends = (next_states == -1).all(axis=1)  
                next_q_values[episode_ends] = 0  
                
                # Target Q-value: r + gamma * Q(s', a')
                target_q_values = rewards_tensor + gamma * next_q_values

            # Current Q-values
            current_q_values = self.q_net(states_tensor, actions_tensor).squeeze(1)
            q_loss = loss_fn(current_q_values, target_q_values)
            q_losses.append(q_loss.item())
            q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            q_optimizer.step()

            # --- Policy Network Update ---
            sampled_actions = self.policy_net(states_tensor)
            perturbations = sampled_actions + self.perturb_net(states_tensor, sampled_actions)
            perturbed_actions = torch.clamp(perturbations, min=-0.06, max=0.35)

            q_value_for_actions = self.q_net(states_tensor, perturbed_actions.detach())
            policy_loss = -q_value_for_actions.mean()  # Negative Q-values for policy optimization
            policy_losses.append(policy_loss.item())

            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            policy_optimizer.step()

            # --- Perturbation Network Update ---
            perturb_loss = loss_fn(perturbed_actions.squeeze(1), actions_tensor) 
            perturb_losses.append(perturb_loss.item())
            perturb_optimizer.zero_grad()
            perturb_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.perturb_net.parameters(), max_norm=1.0)
            perturb_optimizer.step()

            # Optionally print the losses every few iterations
            if iteration % 100 == 0:
                print(f'Iteration: {iteration}, Q Loss: {q_loss.item()*100:.4f}, Policy Loss: {policy_loss.item():.4f}')
        
        plot_loss_progress(q_losses, policy_losses, perturb_losses, labels=["Q Loss", "Policy Loss", "Perturb Loss"])
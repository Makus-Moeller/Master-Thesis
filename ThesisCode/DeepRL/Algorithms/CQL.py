import sys
import numpy as np
sys.path.insert(0, 'C:/Udvikler/Speciale/ThesisCode')
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
        x = F.relu(self.fc1(x_input))  # Using ReLU instead of Tanh
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)  # Directly outputting Q-values
        return x

class CQL(): 
    def __init__(self,
                 state_size, 
                 action_size, 
                 hidden_size, 
                 alpha=1.0,
                 seed=42,
                 sample_method="random"):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.set_random_seed(seed)
        self.alpha = alpha  # Regularization strength for conservative loss, greater value greater penalty
        self.sample_method = sample_method
        self.q_net = QNetworkSkipConnection(state_size, action_size, hidden_size)

    def sample_batch(self, experience_data, batch_size):
        '''data should be a list of SARS tuples'''
        idx = np.random.choice(len(experience_data), batch_size, replace=False)
        return [experience_data[i] for i in idx]
    

    def sample_batch_time_bias(self, experience_data, batch_size, chunk_size=128, priority_factor=0.8):
        """
        Sample a mini-batch from experience data with a bias towards more recent chunks.

        Args:
        - experience_data: List of SARS tuples.
        - batch_size: Number of samples in the mini-batch.
        - chunk_size: Number of experiences in each chunk.
        - priority_factor: Factor controlling the priority of recent chunks (0 to 1).
                            Higher values give more weight to recent chunks.

        Returns:
        - A mini-batch list of SARS tuples.
        """
        # Divide experience_data into chunks
        num_chunks = len(experience_data) // chunk_size
        chunks = [experience_data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
        sample_size = chunk_size // 2

        # Assign sampling probabilities to chunks
        weights = np.array([priority_factor ** (num_chunks - i - 1) for i in range(num_chunks)])
        weights /= weights.sum()  # Normalize to get probabilities

        # Sample a chunk based on weighted probabilities
        chosen_chunks = np.random.choice(num_chunks, size=batch_size // chunk_size, p=weights, replace=True)
        
        # Sample experiences from selected chunks to form the mini-batch
        batch = []
        for chunk_idx in chosen_chunks:
            chunk = chunks[chunk_idx]
            sampled_indices = np.random.choice(len(chunk), sample_size, replace=False)
            batch.extend([chunk[i] for i in sampled_indices])
            
        return batch[:batch_size]  


    def set_random_seed(self, seed):
        print("seed in CQL: ", seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(self, experience_data, train_iterations=1000, batch_size=128, lr=0.001, gamma=0.99, 
              optimizer=None, loss_fn=None, follow_progress=True, save_path="q_net_final.pth"):
        print(self.q_net)
        optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr) if optimizer == None else optimizer
        loss_fn = torch.nn.MSELoss() if loss_fn == None else loss_fn
        q_losses = []
        cql_losses = []
        total_losses = []

        for iteration in range(train_iterations):  
            
            # Sample a mini-batch
            batch = self.sample_batch(experience_data, batch_size) if self.sample_method == "random" else self.sample_batch_time_bias(experience_data, batch_size)   # Memory is assumed to be pre-filled with experience tuples
            states = np.array([each[0] for each in batch], dtype=np.float32)
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch], dtype=np.float32)
            next_states = np.array([each[3] for each in batch], dtype=np.float32)

            # Convert to tensors
            next_states_tensor = torch.as_tensor(next_states)  
            rewards_tensor = torch.as_tensor(rewards)
            states_tensor = torch.as_tensor(states)
            actions_tensor = torch.as_tensor(actions, dtype=torch.int64)
            
            # Compute Q values for the next state without gradient
            with torch.no_grad():
                target_Qs_tensor = self.q_net(next_states_tensor)
                
                # Set target Qs to 0 for the terminal state 
                episode_ends = (next_states == -1).all(axis=1) 
                target_Qs_tensor[episode_ends] = torch.zeros(self.action_size)
                
                # Compute the target
                targets_tensor = rewards_tensor + gamma * torch.max(target_Qs_tensor, dim=1)[0]

            # Compute the Q values for the actions taken
            output_tensor = self.q_net(states_tensor)  
            Q_tensor = torch.gather(output_tensor, 1, actions_tensor.unsqueeze(-1)).squeeze() 

            #### Conservative Loss Start ####
            conservative_penalty = torch.logsumexp(output_tensor, dim=1).mean()  

            #Reward Q-values for the actions in the dataset
            dataset_value = Q_tensor.mean()  

            # Conservative loss term 
            cql_loss = self.alpha * (conservative_penalty - dataset_value)
            cql_losses.append(cql_loss.item())
            
            # Compute Bellman error
            q_loss = loss_fn(Q_tensor, targets_tensor)
            q_losses.append(q_loss.item())

            total_loss = q_loss + cql_loss
            total_losses.append(total_loss.item())

            # Gradient-based update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if follow_progress:
            plot_loss_progress(q_losses, cql_losses, total_losses, labels=["Q-Learning Loss", "CQL Loss", "Total Loss"])

        # Save the final model
        torch.save(self.q_net.state_dict(), save_path)
        print(f"Model saved to {save_path}")
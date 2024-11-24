import sys
import numpy as np
sys.path.insert(0, 'C:/Udvikler/Speciale/SpecialeKode')
from model_evaluation.eval_utilities import plot_loss_progress, plot_q_target_vs_prediction
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworkSkipConnectionMarkovGame(nn.Module):
    def __init__(self, state_size, action_size, opponent_action_size, hidden_size, bias=True):
        super(QNetworkSkipConnectionMarkovGame, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size, bias)  
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias)  
        # The output layer now needs to output Q-values for both player and opponent actions
        self.output_layer = nn.Linear(hidden_size + state_size, action_size * opponent_action_size, bias)  # Flattened output

        # Store action sizes for reshaping later
        self.action_size = action_size
        self.opponent_action_size = opponent_action_size

    def forward(self, x_input):
        x = F.tanh(self.fc1(x_input))
        x = F.tanh(self.fc2(x))
        x = torch.cat((x_input, x), dim=1)
        x = self.output_layer(x)

        # Reshape the output to be (batch_size, action_size, opponent_action_size)
        x = x.view(-1, self.action_size, self.opponent_action_size)
        return x

class QNetworkMarkovGame(nn.Module):
    def __init__(self, state_size, action_size, opponent_action_size, hidden_size, bias=True):
        super(QNetworkMarkovGame, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size, bias)  
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias)  
        # The output layer now needs to output Q-values for both player and opponent actions
        self.output_layer = nn.Linear(hidden_size, action_size * opponent_action_size, bias)  # Flattened output

        # Store action sizes for reshaping later
        self.action_size = action_size
        self.opponent_action_size = opponent_action_size

    def forward(self, x_input):
        x = F.relu(self.fc1(x_input))  # Using ReLU activation
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)

        # Reshape the output to be (batch_size, action_size, opponent_action_size)
        x = x.view(-1, self.action_size, self.opponent_action_size)
        return x


class CQLMarkovGame():
    def __init__(self,
                 state_size, 
                 action_size, 
                 opponent_action_size,  # Add opponent's action space size
                 hidden_size, 
                 alpha=1.0,
                 seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.opponent_action_size = opponent_action_size  # Save the size of the opponent's action space
        self.hidden_size = hidden_size
        self.set_random_seed(seed)
        self.alpha = alpha  # Regularization strength for conservative loss
        self.q_net = QNetworkSkipConnectionMarkovGame(state_size, action_size, opponent_action_size, hidden_size)  # Updated Q-network

    def sample_batch(self, experience_data, batch_size):
        '''data should be a list of SARS tuples'''
        idx = np.random.choice(len(experience_data), batch_size, replace=False)
        return [experience_data[i] for i in idx]

    def set_random_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(self, experience_data, train_iterations=1000, batch_size=128, lr=0.001, gamma=0.99, optimizer=None, loss_fn=None):
        print(self.q_net)
        optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr) if optimizer is None else optimizer
        loss_fn = torch.nn.MSELoss() if loss_fn is None else loss_fn
        q_losses = []
        cql_losses = []
        total_losses = []

        for iteration in range(train_iterations):  # Training loop
            # Sample a mini-batch from the dataset
            batch = self.sample_batch(experience_data, batch_size)
            states = np.array([each[0] for each in batch], dtype=np.float32)
            actions = np.array([each[1] for each in batch])  # Player 1's actions
            opponent_actions = np.array([each[2] for each in batch])  # Player 2's actions
            rewards = np.array([each[3] for each in batch], dtype=np.float32)
            next_states = np.array([each[4] for each in batch], dtype=np.float32)

            # Convert to tensors
            next_states_tensor = torch.as_tensor(next_states)
            rewards_tensor = torch.as_tensor(rewards)
            states_tensor = torch.as_tensor(states)
            actions_tensor = torch.as_tensor(actions, dtype=torch.int64)
            opponent_actions_tensor = torch.as_tensor(opponent_actions, dtype=torch.int64)  # Opponent's actions
            
            # Compute Q values for the next state (without gradient)
            with torch.no_grad():
                target_Qs_tensor = self.q_net(next_states_tensor)
                
                # Set target Qs to 0 for terminal states (episode ends)
                episode_ends = (next_states == -1).all(axis=1)  # Identifying terminal states
                target_Qs_tensor[episode_ends] = torch.zeros(self.action_size, self.opponent_action_size)
                
                # Compute the target: r + gamma * max(Q(s', a1', a2'))
                # Compute the max over both player and opponent actions for the next state's Q-values
                max_future_Q = torch.max(torch.max(target_Qs_tensor, dim=2)[0], dim=1)[0]  # Max over both actions
                targets_tensor = rewards_tensor + gamma * max_future_Q

            # Compute the Q values for the current state-action pairs (player + opponent)
            output_tensor = self.q_net(states_tensor)  # Get Q-values for all actions (player + opponent)
            print(output_tensor.shape)
            print(actions_tensor)
            print(opponent_actions_tensor)
            print(actions_tensor.unsqueeze(1).unsqueeze(2)[3,0,0])
            print(output_tensor[:,51,:])
            # Gather Q-values for player actions
            Q_player_tensor = output_tensor.gather(1, actions_tensor.unsqueeze(1).unsqueeze(2))  # Shape: [batch_size, 1, opponent_action_size]
            
            print('Qplayer tensor: ', Q_player_tensor)

            Q_tensor = output_tensor.gather(1, actions_tensor.unsqueeze(1).unsqueeze(2))
            #print(Q_tensor.shape)
            # Gather Q-values for opponent actions
            Q_tensor = Q_tensor.gather(1, opponent_actions_tensor.unsqueeze(1)).squeeze(1)

            #### Conservative Loss Start ####
            # CQL conservative term: Penalize Q-values for policy's actions (max across all actions of both players)
            conservative_penalty = torch.logsumexp(output_tensor, dim=(1, 2)).mean()  # LogSumExp for both actions

            # CQL dataset term: Reward Q-values for the actions in the dataset
            dataset_value = Q_tensor.mean()  # For actions in dataset (actions_tensor for both players)

            # Conservative loss term (maximize dataset actions, minimize policy actions)
            cql_loss = self.alpha * (conservative_penalty - dataset_value)
            cql_losses.append(cql_loss.item())
            #### Conservative Loss End ####

            # Compute standard loss between predicted Q-values and target Q-values
            q_loss = loss_fn(Q_tensor, targets_tensor)
            q_losses.append(q_loss.item())

            # Combine Q-learning loss and conservative loss
            total_loss = q_loss + cql_loss
            total_losses.append(total_loss.item())

            # Gradient-based update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print(f'Iteration: {iteration}, Loss: {q_loss.item():.4f}, CQL Loss: {cql_loss.item():.4f}, Total Loss: {total_loss.item():.4f}')

        plot_loss_progress(q_losses, cql_losses, total_losses, labels=["Q-Learning Loss", "CQL Loss", "Total Loss"])

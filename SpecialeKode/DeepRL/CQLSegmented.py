from collections import Counter
import sys
import numpy as np
sys.path.insert(0, 'C:/Udvikler/Speciale/SpecialeKode')
from model_evaluation.eval_utilities import plot_loss_progress, plot_q_target_vs_prediction
import torch
import torch.nn as nn
import torch.nn.functional as F
from DeepRL.CQL import CQL
from utils import map_int_to_spread_tensor
from scipy.stats import gaussian_kde
    
class CQLSegmented(CQL):
    def __init__(self, state_size, action_size, hidden_size, alpha=1.0, seed=42, sample_method="random", segment_size=128, overlap=10):
        super().__init__(state_size, action_size, hidden_size, alpha, seed, sample_method)
        self.segment_size = segment_size
        self.overlap = overlap

    def estimate_mean_field(self, segment):
        """ Estimate mean-field distribution over primary state index (s[0]). """
        state_counts = Counter(s[0] for s, _, _, _, _ in segment)
        total_states = sum(state_counts.values())
        mean_field = {state: count / total_states for state, count in state_counts.items()}
        return mean_field
    
    def estimate_mean_field_continuous(self, segment):
        """
        Estimate the mean field by calculating the mean of continuous state features 
        over the entire segment.

        Parameters:
        - segment: List of tuples (state, action, next_state, reward_terms)

        Returns:
        - mean_field: A list or array representing the average state across the segment
        """
        # Extract all states from the segment
        states = np.array([s[0] for s in segment])

        # Compute the mean for each feature in the state vector
        mean_state = np.mean(states, axis=0)
        
        return mean_state
    


    def sample_inventory_distribution(self, segment, batch_size):
        """
        Estimate a KDE over inventory states for a given segment.
        
        Parameters:
        - segment: List of tuples (state, action, next_state, reward_terms)
        
        Returns:
        - kde: A Gaussian KDE object representing the inventory distribution
        """
        # Extract the inventory states from each state in the segment
        inventories = np.array([s[0] for s in segment])  # Assuming inventory is the first feature in state
        
        print(inventories)

        # Fit a KDE to the inventory data
        kde = gaussian_kde(inventories, bw_method='scott')  # Adjust bandwidth if needed
        
        inventory_samples = kde.resample(batch_size).reshape(-1)

        return inventory_samples


    def calculate_rewards(self, actions, mu_actions, reward_terms, reward_function, alpha=1.0):
        """
        Calculate the reward based on RFQ outcome, inventory changes, and deviation from the most likely mean-field action.
        
        Parameters:
        - states: current state
        - actions: action taken
        - mu_actions: actions based on meanfield
        - reward_terms: the reward paramters needed to calculate the reward conditioned on mean field
        - alpha: penalty coefficient for deviation from the most likely mean-field policy
        
        Returns:
        - Reward (modified based on \(\pi(\mu)\) deviation)
        """
        
        if reward_function.__name__ == 'risk_PnL_reward':
            #Read reward parameters 
            mean_spread_PnLs = torch.tensor([term[0] for term in reward_terms]) 
            amounts = torch.tensor([term[1] for term in reward_terms])
            inventory_PnLs = torch.tensor([term[2] for term in reward_terms])
            inventory_penalties = torch.tensor([term[3] for term in reward_terms]) 
            
            #Convert actions to spreads
            spreads = map_int_to_spread_tensor(actions, -0.005, 0.15) #OBS hardcoded, need acess to datatransformer otherwise
            mu_spreads = map_int_to_spread_tensor(mu_actions, -0.005, 0.15) 

            #Calculate indicator variable    
            won_t = torch.where(spreads <= mu_spreads, torch.tensor(1.0), torch.tensor(0.0))  
            spreadPnL = won_t*amounts*spreads    

            # Deviation penalty based on difference from most likely inventory level
            deviation_penalty = alpha * torch.abs(spreads - mu_spreads)
            
            # Total reward considering deviation from the mean-field most likely inventory
            total_rewards =  spreadPnL / mean_spread_PnLs + inventory_PnLs - inventory_penalties - deviation_penalty
        
        elif reward_function.__name__ == 'inventory_PnL_reward':
            #Read reward parameters 
            mean_spread_PnLs = torch.tensor([term[0] for term in reward_terms]) 
            amounts = torch.tensor([term[1] for term in reward_terms])
            inventory_PnLs = torch.tensor([term[2] for term in reward_terms])
            inventoryRisks = torch.tensor([term[3] for term in reward_terms])
            directions = torch.tensor([term[4] for term in reward_terms])
            std_inventory_risks = torch.tensor([term[5] for term in reward_terms]) 
            
            #Convert actions to spreads
            spreads = map_int_to_spread_tensor(actions, -0.005, 0.15) #OBS hardcoded, need acess to datatransformer otherwise
            mu_spreads = map_int_to_spread_tensor(mu_actions, -0.005, 0.15) 

            #Calculate indicator variable    
            won_t = torch.where(spreads <= mu_spreads, torch.tensor(1.0), torch.tensor(0.0))  
            spreadPnL = won_t*amounts*spreads    

            inventory_penalties = ((inventoryRisks - won_t * directions * amounts)/std_inventory_risks**2)/2 

            # Deviation penalty based on difference from most likely inventory level
            deviation_penalty = alpha * torch.abs(spreads - mu_spreads)
            
            # Total reward considering deviation from the mean-field most likely inventory
            total_rewards =  spreadPnL / mean_spread_PnLs + inventory_PnLs - inventory_penalties - deviation_penalty 
        
        else: raise ValueError(f"Unsupported reward function: {reward_function.__name__}")  
        
        return total_rewards

    def mean_field_policy(self, contextual_states_tensor):
        """
        Return the action based on the most likely mean-field policy plus context \(\pi(\mu+context)\).
        """
        # Placeholder: map most likely inventory level to a suggested action
        # Get Q-values for the state
        with torch.no_grad():  # Ensure no gradients are calculated
            Qs = self.q_net(contextual_states_tensor)
            # Select the action with the highest Q-value (policy)
            best_actions = torch.argmax(Qs, dim=1)

        return best_actions  # In practice, define this mapping based on your action strategy

    def train_with_segments(self, experience_data, reward_function, train_iterations=20, policy_iterations=40, batch_size=64, lr=0.001, gamma=0.99, follow_progress=True):
        # Initialize optimizer and loss function
        optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        q_losses, cql_losses, total_losses = [], [], []

        for iteration in range(train_iterations):
            # Loop through each segment
            for start in range(0, len(experience_data) - self.segment_size, self.segment_size - self.overlap):
                # Extract the current segment
                segment = experience_data[start: start + self.segment_size]
                
                # Estimate mean-field distribution for this segment
                mean_field = self.estimate_mean_field(segment)

                # Perform policy updates for this segment
                for j in range(policy_iterations):
                    batch = self.sample_batch(segment, batch_size)
                    
                    states = np.array([each[0] for each in batch], dtype=np.float32)
                    actions = np.array([each[1] for each in batch])
                    next_states = np.array([each[3] for each in batch], dtype=np.float32)
                    reward_terms = np.array([each[4] for each in batch], dtype=np.float32)
                    
                    # Convert to tensors
                    states_tensor = torch.as_tensor(states)
                    actions_tensor = torch.as_tensor(actions, dtype=torch.int64)
                    next_states_tensor = torch.as_tensor(next_states)

                    # Calculate contextual state [mu, s2, s3, s4]
                    most_likely_inventory = max(mean_field, key=mean_field.get)  # Most likely state
                    meanfield_states = [
                        [most_likely_inventory] + s[1:].tolist() for s in states
                    ]
                    
                    meanfield_states_tensor = torch.as_tensor(meanfield_states)
                    
                    mu_actions = self.mean_field_policy(meanfield_states_tensor) #Return tensor
                    
                    rewards_tensor = self.calculate_rewards(actions_tensor, mu_actions, reward_terms, reward_function)
                    #rewards_tensor = torch.as_tensor(rewards)

                    # Compute Q values for the next state (without gradient)
                    with torch.no_grad():
                        target_Qs_tensor = self.q_net(next_states_tensor)
                        episode_ends = (next_states == -1).all(axis=1)
                        target_Qs_tensor[episode_ends] = torch.zeros(self.action_size)
                        targets_tensor = rewards_tensor + gamma * torch.max(target_Qs_tensor, dim=1)[0]

                    # Compute Q values for the selected actions
                    output_tensor = self.q_net(states_tensor)
                    Q_tensor = torch.gather(output_tensor, 1, actions_tensor.unsqueeze(-1)).squeeze()

                    #### Conservative Loss Calculation ####
                    conservative_penalty = torch.logsumexp(output_tensor, dim=1).mean()
                    dataset_value = Q_tensor.mean()
                    cql_loss = self.alpha * (conservative_penalty - dataset_value)
                    cql_losses.append(cql_loss.item())

                    # Compute the Q-learning loss
                    q_loss = loss_fn(Q_tensor, targets_tensor)
                    q_losses.append(q_loss.item())

                    # Combine the Q-learning loss and CQL loss
                    total_loss = q_loss + cql_loss
                    total_losses.append(total_loss.item())

                    # Gradient-based update
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # Optionally print progress
                    #if follow_progress and j % 10 == 0:
                    #    print(f'Iteration: {iteration}, Segment Step: {j}, Q Loss: {q_loss.item()}, CQL Loss: {cql_loss.item()}')
            
            # Optional: plot or save progress after each segment loop
        if follow_progress:
            plot_loss_progress(q_losses, cql_losses, total_losses, labels=["Q Loss", "CQL Loss", "Total Loss"])

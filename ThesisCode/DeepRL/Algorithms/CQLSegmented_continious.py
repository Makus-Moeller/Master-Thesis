from collections import Counter
import sys
import numpy as np
sys.path.insert(0, 'C:/Udvikler/Speciale/ThesisCode')
from model_evaluation.eval_utilities import plot_loss_progress
import torch
import torch.nn.functional as F
from DeepRL.Algorithms.CQL import CQL
from utils import map_int_to_spread_tensor
from scipy.stats import gaussian_kde
    
class CQLSegmented(CQL):
    def __init__(self, state_size, action_size, hidden_size, alpha=1.0, seed=42, sample_method="random", segment_size=128, overlap=10):
        super().__init__(state_size, action_size, hidden_size, alpha, seed, sample_method)
        self.segment_size = segment_size
        self.overlap = overlap


    def sample_inventory_distribution(self, segment, batch_size):
        """
        Estimate a KDE over inventory states for a given segment.
        
        Parameters:
        - segment: List of tuples (state, action, next_state, reward_terms)
        
        Returns:
        - kde: A Gaussian KDE object representing the inventory distribution
        """
        # Extract the inventory states from each state in the segment
        inventories = np.array([s[0] for s,_,_,_,_ in segment])  

        # Fit a KDE to the inventory data
        kde = gaussian_kde(inventories, bw_method='scott')  
        inventory_samples = kde.resample(batch_size).reshape(-1)

        return inventory_samples



    def calculate_rewards(self, actions, mu_actions, reward_terms, reward_function, beta=1.0, dist_fn='abs', w1=1.0, w2=1.0, w3=0.5):
        """
        Calculate the reward based on RFQ outcome, inventory changes, and deviation from the most likely mean-field action.
        """
        # Extract reward parameters once
        mean_spread_PnLs = torch.tensor([term[0] for term in reward_terms]) 
        std_spread_PnLs = torch.tensor([term[1] for term in reward_terms])
        amounts = torch.tensor([term[2] for term in reward_terms])
        inventory_PnLs_standard = torch.tensor([term[3] for term in reward_terms])
        directions = torch.tensor([term[4] for term in reward_terms])

        # Convert actions to spreads
        spreads = map_int_to_spread_tensor(actions, -0.005, 0.15)
        mu_spreads = map_int_to_spread_tensor(mu_actions, -0.005, 0.15)

        # Calculate indicator variable and spread PnL
        won_t = torch.where(spreads <= mu_spreads, torch.tensor(1.0), torch.tensor(0.0))
        spreadPnL = won_t * amounts * spreads
        spreadPnL_standard = (spreadPnL - mean_spread_PnLs) / std_spread_PnLs

        # Distance penalty
        if dist_fn == "abs":
            dist = torch.abs(spreads - mu_spreads)
        elif dist_fn == "quadratic":
            dist = (spreads - mu_spreads)**2
        else:
            epsilon = 1e-8  
            dist = torch.abs(spreads - mu_spreads)/(torch.abs(mu_spreads)+epsilon)
        
        # Normalize deviation to [0, 1] range by dividing by the maximum deviation in the batch
        max_dist = dist.max().item() if dist.max().item() > 0 else 1  
        dist_normalized = dist / max_dist
        deviation_penalty = beta * dist_normalized

        # Calculate risk/inventory penalty and total rewards based on the reward function
        if reward_function.__name__ == 'risk_PnL_reward':
            bpvRisks = torch.tensor([term[5] for term in reward_terms])
            std_bpv_risks = torch.tensor([term[6] for term in reward_terms])
            mean_bpv_risks = torch.tensor([term[7] for term in reward_terms])
            bpvs = torch.tensor([term[8] for term in reward_terms])
            bpvRisk_penalties_stand = ((bpvRisks - (won_t * directions * amounts * bpvs) / 1000) - mean_bpv_risks) / std_bpv_risks
            total_rewards = w1*spreadPnL_standard + w2*inventory_PnLs_standard - w3*bpvRisk_penalties_stand - deviation_penalty
        
        elif reward_function.__name__ == 'inventory_PnL_reward':
            inventoryRisks = torch.tensor([term[5] for term in reward_terms])
            std_inventory_risks = torch.tensor([term[6] for term in reward_terms])
            mean_inventory_risk = torch.tensor([term[7] for term in reward_terms])
            inventory_penalties_stand = ((inventoryRisks - (won_t * directions * amounts) / 1000) - mean_inventory_risk) / std_inventory_risks
            total_rewards = w1*spreadPnL_standard + w2*inventory_PnLs_standard - inventory_penalties_stand - deviation_penalty
        
        else:
            raise ValueError(f"Unsupported reward function: {reward_function.__name__}")

        return total_rewards


    def mean_field_policy_random(self, contextual_states_tensor, tau=1.0):
        contextual_states_tensor = contextual_states_tensor.float()
    
        with torch.no_grad():
            Qs = self.q_net(contextual_states_tensor)

            if torch.isnan(Qs).any() or torch.isinf(Qs).any():
                print("Warning: Q-values contain NaN or Inf. Fixing them.")
                Qs = torch.nan_to_num(Qs, nan=0.0, posinf=1e6, neginf=-1e6)

            Qs = Qs.clamp(min=-1e6, max=1e6)
            action_probs = F.softmax(Qs / tau, dim=1)
            
            # Check action_probs for NaN or Inf after softmax
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any() or (action_probs < 0).any():
                raise ValueError("action_probs contains invalid values after softmax.")
                 
            sampled_actions = torch.multinomial(action_probs, num_samples=1).squeeze()
        return sampled_actions



    def train_with_segments(self, experience_data, reward_function, train_iterations=20, policy_iterations=40, 
                            batch_size=64, lr=0.001, gamma=0.99, follow_progress=True, dist_fn="abs"):
        # Initialize optimizer and loss function
        optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        q_losses, cql_losses, total_losses = [], [], []

        for iteration in range(train_iterations):
            # Loop through each segment
            for start in range(0, len(experience_data) - self.segment_size, self.segment_size - self.overlap):
                
                segment = experience_data[start: start + self.segment_size]
                
                # Perform policy updates for this segment
                for j in range(policy_iterations):
                    batch = self.sample_batch(segment, batch_size)
                    
                    #Divide batch
                    states = np.array([each[0] for each in batch], dtype=np.float32)
                    actions = np.array([each[1] for each in batch])
                    next_states = np.array([each[3] for each in batch], dtype=np.float32)
                    reward_terms = np.array([each[4] for each in batch], dtype=np.float32)
                    
                    # Convert to tensors
                    states_tensor = torch.as_tensor(states)
                    actions_tensor = torch.as_tensor(actions, dtype=torch.int64)
                    next_states_tensor = torch.as_tensor(next_states)
                    
                    #Make meanfield and meanfield actions
                    mean_field_sample = self.sample_inventory_distribution(segment, batch_size)
                    mean_field_sample_tensor = torch.tensor(mean_field_sample, dtype=torch.float32).unsqueeze(1)  

                    context_features_tensor = torch.tensor(states[:, 1:], dtype=torch.float32)  
                    meanfield_states_tensor = torch.cat((mean_field_sample_tensor, context_features_tensor), dim=1)
                    mu_actions = self.mean_field_policy_random(meanfield_states_tensor, tau=1) 
                    
                    rewards_tensor = self.calculate_rewards(actions_tensor, mu_actions, reward_terms, reward_function, dist_fn=dist_fn)
                    
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

                   
        if follow_progress:
            plot_loss_progress(q_losses, cql_losses, total_losses, labels=["Q Loss", "CQL Loss", "Total Loss"])

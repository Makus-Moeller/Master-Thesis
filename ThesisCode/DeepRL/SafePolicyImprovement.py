import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from scipy.stats import norm


gamma = 0.99  


def estimate_behavior_policy(dataset, bandwidth=0.5, epsilon=1e-6):
    """Estimate the behavior policy using KDE with numerical stability."""
    states = np.array([d[0] for d in dataset])
    actions = np.array([d[1] for d in dataset]).reshape(-1, 1)
    
    # Joint density for (state, action)
    kde_joint = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.hstack((states, actions)))
    
    # Marginal density for states
    kde_state = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(states)
    
    def behavior_policy(state, action):
        # Convert state to NumPy array
        state = np.array(state)
        
        # Compute joint and state densities
        joint_density = np.exp(kde_joint.score_samples(np.hstack((state.reshape(1, -1), [[action]]))))
        state_density = max(np.exp(kde_state.score_samples(state.reshape(1, -1))), epsilon)  # Add stability
        
        # Compute initial pi_beta
        pi_beta = joint_density / state_density if state_density > 0 else 0
        
        # Normalize across all actions
        all_actions = range(155)  
        normalization_factor = sum(
            max(np.exp(kde_joint.score_samples(np.hstack((state.reshape(1, -1), [[a]])))) / state_density, epsilon)
            for a in all_actions
        )
        pi_beta /= normalization_factor if normalization_factor > 0 else 1

        return pi_beta
    
    return behavior_policy



def learned_policy(q_network, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  
    q_values = q_network(state_tensor).detach().numpy().flatten()

    # Stabilize softmax computation
    max_q = np.max(q_values)
    scaled_q_values = q_values - max_q  # Subtract max Q-value
    exp_q_values = np.exp(scaled_q_values / 10)  
    softmax_probs = exp_q_values / np.sum(exp_q_values)

    return softmax_probs


# Compute penalty term
def confidence_adjustment(delta, sample_size):
    """Compute the confidence adjustment for a given delta and sample size."""
    z = norm.ppf(1 - delta / 2)  
    return z / np.sqrt(sample_size)

def cql_penalty(q_network, behavior_policy, state, delta, dataset_size):
    """Compute the CQL penalty with a threshold for pi_beta."""
    actions = range(q_network.output_layer.out_features)  # Assuming discrete actions
    pi_star = learned_policy(q_network, state)
    penalty = 0
    for a in actions:
        pi_beta = behavior_policy(state, a)
        pi_beta = max(pi_beta, 1e-8)
        if pi_beta > 0:
            penalty_term = min(pi_star[a] * (pi_star[a] / pi_beta - 1), 1e6)
            penalty += penalty_term + confidence_adjustment(delta, dataset_size)
    return penalty

# Compute sampling error
def sampling_error(q_network, behavior_policy, dataset, delta):
    penalties = []
    for state, _, _ in dataset:
        penalty = cql_penalty(q_network, behavior_policy, state, delta, len(dataset))
        penalties.append(np.sqrt(q_network.output_layer.out_features) * np.sqrt(penalty + 1))
    
    mean_penalty = np.mean(penalties)
    confidence_bound = norm.ppf(1 - delta / 2) / np.sqrt(len(dataset))
    sampling_error = (gamma / (1 - gamma)**2) * (mean_penalty + confidence_bound)
    return sampling_error

# Compute empirical returns
def empirical_return(dataset, gamma):
    """Compute the empirical return for a given dataset."""
    returns = []
    for _, _, reward in dataset:  # Assuming dataset includes rewards
        returns.append(reward / (1 - gamma))
    return np.mean(returns)

# Compute \zeta
def compute_zeta(q_network, dataset_train, dataset_val, delta):
    """Compute the \zeta value with confidence bounds."""
    
    behavior_policy = estimate_behavior_policy(dataset_train, bandwidth=1.0)
    sampling_error_term = sampling_error(q_network, behavior_policy, dataset_val, delta)
    print(f"Sampling Error Term: {sampling_error_term}")
    
    empirical_returns = empirical_return(dataset_val, gamma)
    print(f"Empirical Returns: {empirical_returns}")
    
    # Compute zeta
    zeta = sampling_error_term - empirical_returns
    print(f"Zeta: {zeta}")
    return zeta
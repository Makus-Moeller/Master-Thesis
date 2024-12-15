import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pickle
from utils import * 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils import shuffle




##################### Alpha Functions #######################################################

#Alpha functions: Have to satisfy Robbins-Monro conditions
def alpha_1(N_t): #nt number of times action a is taken in given state 
    return 2/((N_t**(2/3))+1)

def alpha_2(t): #nt number of times action a is taken in given state 
    return 1/(t+1)


################### MDP learning algorithms ####################################################

def QL(D, gamma, Q):
    '''
    Implementation of OPO Q-learning.

    :param data: Trajectory we want to learn from 
    :param gamma: Discount factor when using discounted MDPs
    :param Q: Random initialized Q-table  

    returns: np.ndarray 
    '''
  
    for t in range(len(D)-1):

        s, a, r, ns = int(D.iloc[t]['State_index']), int(D.iloc[t]['Action']), D.iloc[t]['Reward'], int(D.iloc[t]['Next_State'])
        alpha = alpha_2(t+1)
        max_next_Q = np.max(Q[ns, :]) if ns else 0 #raise ValueError("Next state not found")?
        delta_t = r + gamma * max_next_Q - Q[s, a]
        Q[s, a] += alpha * delta_t

    optimal_policy = np.argmax(Q, axis=1)
    return optimal_policy

def QL(D, gamma, num_states, num_actions):
    '''
    Implementation of OPO Q-learning.

    :param data: Trajectory we want to learn from 
    :param gamma: Discount factor when using discounted MDPs
    :param num_states: Number of states, used to initialize Q-table 
    :param num_actions: Number of actions, used to initialize Q-table 

    returns: np.ndarray 
    '''
    #Make Q-table
    Q = np.zeros((num_states, num_actions), dtype=np.float32)

    # Calculate the most frequent action in the dataset
    most_frequent_action = int(D['Action'].mode()[0])
    print("most frequent action: ", most_frequent_action)

    for t in range(len(D)-1):
        s, a, r, ns = int(D.iloc[t]['State_index']), int(D.iloc[t]['Action']), D.iloc[t]['Reward'], int(D.iloc[t]['Next_State'])
        alpha = alpha_2(t+1)
        max_next_Q = np.max(Q[ns, :]) if ns else 0
        delta_t = r + gamma * max_next_Q - Q[s, a]
        Q[s, a] += alpha * delta_t

    
    # Generate the optimal policy
    optimal_policy = []
    for state in range(num_states):
        if np.all(Q[state, :] == 0):  
            optimal_policy.append(most_frequent_action) 
        else:
            optimal_policy.append(np.argmax(Q[state, :]))  

    return np.array(optimal_policy)



def double_QL(D, gamma, QA, QB):
    """
    Implementation of Double Q-learning.

    Args:
        D (pd.DataFrame): Trajectory we want to learn from.
        gamma (float): Discount factor when using discounted MDPs.
        QA (np.ndarray): First Q-table.
        QB (np.ndarray): Second Q-table.

    Returns:
        np.ndarray: Optimal policy based on Double Q-learning.
    """
    for t in range(len(D) - 1):
        s, a, r, ns = int(D.iloc[t]['State_index']), int(D.iloc[t]['Action']), D.iloc[t]['Reward'], int(D.iloc[t]['Next_State'])
        alpha = alpha_2(t + 1)
        
        # Randomly choose which Q-table to update
        if np.random.rand() < 0.5:
            a_hat = np.argmax(QA[ns, :]) if ns else 0
            delta_t = r + gamma * QB[ns,a_hat] - QA[s, a]
            QA[s, a] += alpha * delta_t
        else:
            a_hat = np.argmax(QB[ns, :]) if ns else 0
            delta_t = r + gamma * QA[ns,a_hat] - QB[s, a]
            QB[s, a] += alpha * delta_t

    # Combine Q1 and Q2 to get the optimal policy
    optimal_policy = np.argmax(QA + QB, axis=1)
    return optimal_policy

def double_QL(D, gamma, num_states, num_actions):
    """
    Implementation of Double Q-learning.

    Args:
        D (pd.DataFrame): Trajectory we want to learn from.
        gamma (float): Discount factor when using discounted MDPs.
        num_states (int): Number of states, used to initialize Q-table 
        num_actions  (int): Number of actions, used to initialize Q-table 
    
    Returns:
        np.ndarray: Optimal policy based on Double Q-learning.
    """

    #Make Q-tables
    QA = np.zeros((num_states, num_actions), dtype=np.float32)
    QB = np.zeros((num_states, num_actions), dtype=np.float32)

    # Calculate the most frequent action in the dataset
    most_frequent_action = int(D['Action'].mode()[0])
    print("most frequent action: ", most_frequent_action)

    for t in range(len(D)-1):
        s, a, r, ns = int(D.iloc[t]['State_index']), int(D.iloc[t]['Action']), D.iloc[t]['Reward'], int(D.iloc[t]['Next_State'])
        alpha = alpha_2(t + 1)
        
        # Randomly choose which Q-table to update
        if np.random.rand() < 0.5:
            a_hat = np.argmax(QA[ns, :]) if ns else 0
            delta_t = r + gamma * QB[ns,a_hat] - QA[s, a]
            QA[s, a] += alpha * delta_t
        else:
            a_hat = np.argmax(QB[ns, :]) if ns else 0
            delta_t = r + gamma * QA[ns,a_hat] - QB[s, a]
            QB[s, a] += alpha * delta_t

    # Combine QA and QB to get the optimal policy
    optimal_policy = []
    combined_Q = QA + QB  # Combine the Q-tables
    
    for state in range(num_states):
        if np.all(combined_Q[state, :] == 0):  # Check if the state row is all zeros
            optimal_policy.append(most_frequent_action)  # Use the most frequent action as fallback
        else:
            optimal_policy.append(np.argmax(combined_Q[state, :]))  # Use the action with the highest Q-value

    return np.array(optimal_policy)


################### Markov game learning algorithms ####################################################


def QL_MarkovGame(D, gamma, num_states, num_actions):
    '''
    Implementation of Q-learning for Player 1 in a two-player Markov game.
    
    :param D: Trajectory we want to learn from, with columns ['State_index', 'Action_1', 'Action_2', 'Reward_1', 'Next_State']
    :param gamma: Discount factor when using discounted MDPs
    :param num_states: Number of states, used to initialize Q-table 
    :param num_actions_1: Number of actions for Player 1, used to initialize Q-table
    :param num_actions_2: Number of actions for Player 2, used to initialize Q-table

    :return: np.ndarray containing the optimal policy for Player 1
    '''
    # Initialize Q-table for Player 1: dimensions are (states, actions of player 1, actions of player 2)
    Q1 = np.zeros((num_states, num_actions, num_actions), dtype=np.float32)

    # Loop through the dataset to update the Q1 table
    for t in range(len(D)-1):
        s = int(D.iloc[t]['State_index'])                 # Current state
        a1 = int(D.iloc[t]['Action'])                     # Player 1's action
        a2 = int(D.iloc[t]['Opponent_Action'])            # Player 2's action
        r1 = D.iloc[t]['Reward']                          # Player 1's reward
        ns = int(D.iloc[t]['Next_State'])                 # Next state
        
        # Learning rate, assuming alpha_2(t+1) is defined elsewhere
        alpha = alpha_2(t+1)                                

        # Compute the maximum future Q1 value for Player 1 given Player 2's current action
        max_next_Q1 = np.max(Q1[ns, :, a2]) if ns else 0

        # Calculate the temporal difference target for Player 1
        delta_t = r1 + gamma * max_next_Q1 - Q1[s, a1, a2]

        # Update the Q1 value
        Q1[s, a1, a2] += alpha * delta_t

    # Derive the optimal policy for Player 1
    # Choose the action for Player 1 that maximizes Q1 over all Player 2's actions
    optimal_policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        # Sum Q-values over all actions of Player 2 and find the best action for Player 1
        optimal_policy[s] = np.argmax(np.sum(Q1[s, :, :], axis=1))

    return optimal_policy


def double_QL_MarkovGame(D, gamma, num_states, num_actions):
    '''
    Implementation of Double Q-learning for Player 1 in a two-player Markov game.
    
    :param D: Trajectory we want to learn from, with columns ['State_index', 'Action_1', 'Action_2', 'Reward_1', 'Next_State']
    :param gamma: Discount factor when using discounted MDPs
    :param num_states: Number of states, used to initialize Q-tables 
    :param num_actions: Number of actions for Player 1 (and Player 2 assumed same), used to initialize Q-tables

    :return: np.ndarray containing the optimal policy for Player 1
    '''
    
    # Initialize two Q-tables for Player 1, with dimensions (states, actions of player 1, actions of player 2)
    QA = np.zeros((num_states, num_actions, num_actions), dtype=np.float32)
    QB = np.zeros((num_states, num_actions, num_actions), dtype=np.float32)

    # Loop through the dataset to update the Q-tables
    for t in range(len(D)-1):
        s = int(D.iloc[t]['State_index'])                 # Current state
        a1 = int(D.iloc[t]['Action'])                     # Player 1's action
        a2 = int(D.iloc[t]['Opponent_Action'])            # Player 2's action
        r1 = D.iloc[t]['Reward']                          # Player 1's reward
        ns = int(D.iloc[t]['Next_State'])                 # Next state

        # Learning rate
        alpha = alpha_2(t+1)

        # Randomly choose which Q-table to update
        if np.random.rand() < 0.5:
            # Use QB to estimate the action for the next state
            a1_hat = np.argmax(QA[ns, :, a2]) if ns else 0
            delta_t = r1 + gamma * QB[ns, a1_hat, a2] - QA[s, a1, a2]
            QA[s, a1, a2] += alpha * delta_t
        else:
            # Use QA to estimate the action for the next state
            a1_hat = np.argmax(QB[ns, :, a2]) if ns else 0
            delta_t = r1 + gamma * QA[ns, a1_hat, a2] - QB[s, a1, a2]
            QB[s, a1, a2] += alpha * delta_t

    # Derive the optimal policy for Player 1
    # We combine the QA and QB tables and find the best action for Player 1
    optimal_policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        # Sum over Player 2's actions and find the best action for Player 1
        optimal_policy[s] = np.argmax(np.sum(QA[s, :, :] + QB[s, :, :], axis=1))

    return optimal_policy

################################### Random Policies ###################################################

def random_policy_from_distribution(D):
    
    action_counts = D["Action"].value_counts().sort_index()  
    
    action_probs = action_counts / action_counts.sum()
    
    def random_policy():
        return np.random.choice(action_counts.index, p=action_probs.values)
    
    return random_policy

def random_policy_from_uniform(D):
    # Get the minimum and maximum action values
    min_action = D["Action"].min()
    max_action = D["Action"].max()
    
    # Define the random policy to uniformly pick actions from the interval [min_action, max_action]
    def random_policy():
        return np.random.randint(min_action, max_action + 1)  # Uniform random action from [min_action, max_action]
    
    return random_policy
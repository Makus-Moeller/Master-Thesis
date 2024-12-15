import numpy as np
import pandas as pd
from utils import * 
from DeepRL.datatransformer import DataTransformerDeep
from algortihms import *
from typing import Type
from typing import Callable
from DeepRL.Algorithms.BCQ import BCQ
from DeepRL.Algorithms.miniBatch_Q import MBQ
from DeepRL.Algorithms.CQL import CQL
from DeepRL.Algorithms.CQLSegmented_continious import CQLSegmented
from TabularRL.RFQAgent import DataTransformer
import torch
import torch.nn.functional as F
from typing import Union

def get_actions_for_validation(mainQN, validation_data):
    """
    Apply the learned policy to the validation states and return the predicted actions.

    :param mainQN: Trained Q-network
    :param validation_data: List of tuples (state, action, reward, next_state)
    :return: List of actions predicted for each state in the validation data
    """
    actions = [] 
    with torch.no_grad():  # Ensure no gradients are calculated
        for state, _, _, _ in validation_data:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            Qs = mainQN(state_tensor)
            best_action = torch.argmax(Qs).item()
            actions.append(best_action)

    return actions

def get_actions_for_validation_randomized(mainQN, validation_data, tau=1.0):
    """
    Apply the learned policy to the validation states and return the predicted actions,
    using a softmax distribution over Q-values.

    :param mainQN: Trained Q-network
    :param validation_data: List of tuples (state, action, reward, next_state)
    :param tau: Temperature parameter for softmax. Higher values increase randomness.
    :return: List of actions sampled from the softmax distribution for each state in the validation data
    """
    actions = []  

    with torch.no_grad():  
        for state, _, _, _ in validation_data:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            Qs = mainQN(state_tensor)
            # Apply softmax to get a probability distribution over actions
            action_probs = F.softmax(Qs / tau, dim=1)
            # Sample an action according to the distribution
            action = torch.multinomial(action_probs, num_samples=1).item()
            actions.append(action)
    
    return actions



def get_actions_for_validation_bcq(bcq_agent, validation_data):
    """
    Apply the learned BCQ policy to the validation states and return the predicted actions.

    :param bcq_agent: Trained BCQ agent containing policy_net and perturb_net
    :param validation_data: List of tuples (state, action, reward, next_state)
    :return: List of actions predicted for each state in the validation data
    """
    actions = []  

    with torch.no_grad():  
        for state, _, _, _ in validation_data:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Get the action proposed by the policy network
            proposed_action = bcq_agent.policy_net(state_tensor)
            
            # Apply perturbation to the proposed action
            perturbed_action = proposed_action + bcq_agent.perturb_net(state_tensor, proposed_action)
            best_action = perturbed_action.squeeze(0).cpu().item()  
            actions.append(best_action)

    return actions


def get_actions_for_validation_MarkovGame(mainQN, validation_data):
    """
    Apply the learned policy to the validation states and return the predicted actions for Player 1 in a Markov game.

    :param mainQN: Trained Q-network for Markov game (Player 1)
    :param validation_data: List of tuples (state, action_1, action_2, reward_1, next_state)
    :return: List of actions predicted for each state in the validation data for Player 1
    """
    actions = []  

    with torch.no_grad(): 
        for state, _, _, _, _ in validation_data:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            Qs = mainQN(state_tensor)
            
            # Sum Q-values across opponent's actions (dim=2) to get effective Q-values for Player 1
            player_Qs = Qs.sum(dim=2)  
            best_action = torch.argmax(player_Qs).item()
            actions.append(best_action)

    return actions
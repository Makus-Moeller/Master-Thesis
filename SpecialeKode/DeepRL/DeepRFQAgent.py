import numpy as np
import pandas as pd
from utils import * 
from DeepRL.datatransformer import DataTransformerDeep
from algortihms import *
from typing import Type
from typing import Callable
from DeepRL.BCQ import BCQ
from DeepRL.miniBatch_Q import MBQ
from DeepRL.CQL import CQL
from DeepRL.CQL_Markov_game import CQLMarkovGame
from DeepRL.CQLSegmented_continious import CQLSegmented
from RFQAgent import DataTransformer
import torch
import torch.nn.functional as F
from algortihms import FQI
from typing import Union

def get_actions_for_validation(mainQN, validation_data):
    """
    Apply the learned policy to the validation states and return the predicted actions.

    :param mainQN: Trained Q-network
    :param validation_data: List of tuples (state, action, reward, next_state)
    :return: List of actions predicted for each state in the validation data
    """
    actions = []  # List to store actions

    with torch.no_grad():  # Ensure no gradients are calculated
        for state, _, _, _ in validation_data:
            # Convert the state to a tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Get Q-values for the state
            Qs = mainQN(state_tensor)

            # Select the action with the highest Q-value (policy)
            best_action = torch.argmax(Qs).item()

            # Append the action to the list
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
    actions = []  # List to store actions

    with torch.no_grad():  # Ensure no gradients are calculated
        for state, _, _, _ in validation_data:
            # Convert the state to a tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Get Q-values for the state
            Qs = mainQN(state_tensor)

            # Apply softmax to get a probability distribution over actions
            action_probs = F.softmax(Qs / tau, dim=1)

            # Sample an action according to the distribution
            action = torch.multinomial(action_probs, num_samples=1).item()

            # Append the action to the list
            actions.append(action)

    return actions



def get_actions_for_validation_bcq(bcq_agent, validation_data):
    """
    Apply the learned BCQ policy to the validation states and return the predicted actions.

    :param bcq_agent: Trained BCQ agent containing policy_net and perturb_net
    :param validation_data: List of tuples (state, action, reward, next_state)
    :return: List of actions predicted for each state in the validation data
    """
    actions = []  # List to store actions

    with torch.no_grad():  # Ensure no gradients are calculated
        for state, _, _, _ in validation_data:
            # Convert the state to a tensor and move it to the appropriate device
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Get the action proposed by the policy network
            proposed_action = bcq_agent.policy_net(state_tensor)

            # Apply perturbation to the proposed action
            perturbed_action = proposed_action + bcq_agent.perturb_net(state_tensor, proposed_action)

            # The perturbed action is considered the final action
            best_action = perturbed_action.squeeze(0).cpu().item()  # Remove batch dimension and convert to numpy

            # Append the action to the list (you may need to discretize if your environment has discrete actions)
            actions.append(best_action)

    return actions


def get_actions_for_validation_MarkovGame(mainQN, validation_data):
    """
    Apply the learned policy to the validation states and return the predicted actions for Player 1 in a Markov game.

    :param mainQN: Trained Q-network for Markov game (Player 1)
    :param validation_data: List of tuples (state, action_1, action_2, reward_1, next_state)
    :return: List of actions predicted for each state in the validation data for Player 1
    """
    actions = []  # List to store Player 1's predicted actions

    with torch.no_grad():  # Ensure no gradients are calculated
        for state, _, _, _, _ in validation_data:
            # Convert the state to a tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Get Q-values for the state: output shape is (1, action_size, opponent_action_size)
            Qs = mainQN(state_tensor)

            # Sum Q-values across opponent's actions (dim=2) to get effective Q-values for Player 1
            player_Qs = Qs.sum(dim=2)  # This reduces the shape to (1, action_size)

            # Select the action with the highest summed Q-value for Player 1
            best_action = torch.argmax(player_Qs).item()

            # Append the action to the list
            actions.append(best_action)

    return actions


class DeepRFQAgent():
    def __init__(
        self,
        data_transformer: Union[DataTransformerDeep, DataTransformer],
        random : bool 

        
    ):
        self._data_transformer = data_transformer
        self._random = random
        self._model = None
        self._random_action_generator = None
        

    def predict(self, data, algorithm='BCQ', extended_dataset=True, calibrated=False, random_policy=False):
        assert self._model is not None or self._random_action_generator is not None, "Model is not trained"
        #Transform validation data
        df, original_indexes = self._data_transformer.transform_dataset(data, apply_train_bins=True)
        experience_data =  list(zip(df['State'], df['Action'], df['Reward'], df['Next_State']))

        #get actions
        if algorithm == 'BCQ':
            spreads = get_actions_for_validation_bcq(self._model, experience_data)
        elif algorithm == 'MBQ' or algorithm == 'CQL':
            if random_policy:
                actions = get_actions_for_validation_randomized(self._model, experience_data, tau=1)
            else: 
                actions = get_actions_for_validation(self._model, experience_data)
            spreads = [map_int_to_spread(x, self._data_transformer.lower_spread_limit, self._data_transformer.upper_spread_limit) for x in actions]
        elif algorithm == 'CQL_MarkovGame':
            actions = get_actions_for_validation_MarkovGame(self._model, experience_data)
            spreads = [map_int_to_spread(x, self._data_transformer.lower_spread_limit, self._data_transformer.upper_spread_limit) for x in actions]
        elif algorithm == 'CQL_Segmented':
            if random_policy:
                actions = get_actions_for_validation_randomized(self._model, experience_data, tau=2)
            else:
                actions = get_actions_for_validation(self._model, experience_data) 
            spreads = [map_int_to_spread(x, self._data_transformer.lower_spread_limit, self._data_transformer.upper_spread_limit) for x in actions]
        elif algorithm in ['random_by_action_distr', 'random_by_uniform']:
            actions = np.array([self._random_action_generator() for _ in range(len(df))])
            spreads = [map_int_to_spread(x, self._data_transformer.lower_spread_limit, self._data_transformer.upper_spread_limit) for x in actions]
            
        else: raise ValueError(f"Unsupported algorithm: {algorithm}")  

        #Transform to spread and add to mid price
        data = data.loc[original_indexes].copy()
        data["PolicySpread"] = spreads
        data["ModelPrice"] = data.apply(_model_price_calibrated if calibrated else _model_price, axis=1).copy()
        return np.asarray(data["ModelPrice"], np.float64), data if extended_dataset else np.asarray(data["ModelPrice"], np.float64)  

        

    def train(self, dataset: pd.DataFrame, algorithm='BCQ', gamma=0.99, num_train_steps=1000, 
              hidden_size=128, batch_size=64, alpha=1.0, follow_progress=True, sample_method="random", dist_fn="abs"):
        
        df, _ = self._data_transformer.transform_dataset(dataset)
        
        if self._data_transformer.opponent_actions: 
            experience_data =  list(zip(df['State'], df['Action'], df['Opponent_Action'], df['Reward'], df['Next_State']))
        
        elif self._data_transformer.reward_terms:
            experience_data =  list(zip(df['State'], df['Action'], df['Reward'], df['Next_State'], df['Reward_Terms']))
        else:    
            experience_data =  list(zip(df['State'], df['Action'], df['Reward'], df['Next_State']))
        state_size = len(df['State'][0])
        action_size = int((self._data_transformer.upper_spread_limit - self._data_transformer.lower_spread_limit)*1000)
            

        if algorithm == 'BCQ':
            action_size = 1
            bcq = BCQ(state_size, action_size, hidden_size)
            bcq.train(experience_data, train_iterations=num_train_steps, gamma=gamma, batch_size=batch_size)
            self._model = bcq
        elif algorithm == 'MBQ':
            mbq = MBQ(state_size, action_size, hidden_size)
            mbq.train(experience_data=experience_data, train_iterations=num_train_steps, batch_size=batch_size)
            self._model = mbq.q_net
        elif algorithm == 'CQL':
            #seed = np.random.randint(200) if self._random else 41  #41 is deafualt so no randomness 
            rng = np.random.default_rng()
            seed = rng.integers(400)
            print("seed in RFQAgent: ", seed)
            cql = CQL(state_size, action_size, hidden_size, alpha=alpha, sample_method=sample_method, seed=seed)
            cql.train(experience_data=experience_data, train_iterations=num_train_steps, batch_size=batch_size, follow_progress=follow_progress)
            self._model = cql.q_net
        elif algorithm == 'CQL_MarkovGame':
            cql_Markov = CQLMarkovGame(state_size, action_size, action_size, hidden_size, alpha=alpha)
            cql_Markov.train(experience_data=experience_data, train_iterations=num_train_steps, batch_size=batch_size)
            self._model = cql_Markov.q_net
        elif algorithm=='CQL_Segmented':
            cql_segmented = CQLSegmented(state_size, action_size, hidden_size, alpha=alpha, sample_method=sample_method, segment_size=512, overlap=128)
            cql_segmented.train_with_segments(experience_data=experience_data, 
                                              reward_function = self._data_transformer.reward_function, 
                                              dist_fn= dist_fn,
                                              train_iterations=1,
                                              policy_iterations=100,
                                              follow_progress=follow_progress)
            self._model = cql_segmented.q_net
        
        elif algorithm == 'random_by_action_distr':
            self._random_action_generator = random_policy_from_distribution(df)
    
        elif algorithm == 'random_by_uniform':
             self._random_action_generator = random_policy_from_uniform(df)
    
        else: raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    
    
def _model_price(row):
    """Gives the model price from: `"Mid"`, `"PolicySpread"`, and `"Side"`"""
    mid = row["Mid"]
    spread = row["PolicySpread"]
    side = row["Side"]
    if side == "BUY":
        return mid - spread
    elif side == "SELL":
        return mid + spread
    else:
        raise ValueError(f"Unknown side {side}")


def _model_price_calibrated(row):
    """Gives the model price from: `"Mid"`, `"PolicySpread"`, and `"Side"`"""
    mid = row["AllQMeanMid"] + row["Mid"]
    spread = row["PolicySpread"]
    side = row["Side"]
    if side == "BUY":
        return mid - spread
    elif side == "SELL":
        return mid + spread
    else:
        raise ValueError(f"Unknown side {side}")
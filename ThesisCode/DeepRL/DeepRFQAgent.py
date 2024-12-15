import numpy as np
import pandas as pd
from utils import * 
from DeepRL.datatransformer import DataTransformerDeep
from algortihms import *
from DeepRL.Algorithms.BCQ import BCQ
from DeepRL.Algorithms.miniBatch_Q import MBQ
from DeepRL.Algorithms.CQL import CQL
from DeepRL.Algorithms.CQLSegmented_continious import CQLSegmented
from TabularRL.RFQAgent import DataTransformer
from DeepRL.ExtractPolicies import *
from typing import Union


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
            rng = np.random.default_rng()
            seed = rng.integers(400)
            cql = CQL(state_size, action_size, hidden_size, alpha=alpha, sample_method=sample_method, seed=seed)
            cql.train(experience_data=experience_data, train_iterations=num_train_steps, batch_size=batch_size, follow_progress=follow_progress)
            self._model = cql.q_net
        elif algorithm=='CQL_Segmented':
            cql_segmented = CQLSegmented(state_size, action_size, hidden_size, alpha=alpha, sample_method=sample_method, segment_size=512, overlap=128)
            cql_segmented.train_with_segments(experience_data=experience_data, 
                                              reward_function = self._data_transformer.reward_function, 
                                              dist_fn= dist_fn,
                                              train_iterations=1,
                                              policy_iterations=1000,
                                              follow_progress=follow_progress
                                             )
            self._model = cql_segmented.q_net
        
        elif algorithm == 'random_by_action_distr':
            self._random_action_generator = random_policy_from_distribution(df)
    
        elif algorithm == 'random_by_uniform':
             self._random_action_generator = random_policy_from_uniform(df)
    
        else: raise ValueError(f"Unsupported algorithm: {algorithm}")
    


 ############################ Helpers to add spread to mid price ######################################   
    
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
    """Gives the model price from: `"AllQ Mid"`, `"PolicySpread"`, and `"Side"`"""
    mid = row["AllQMeanMid"] + row["Mid"]
    spread = row["PolicySpread"]
    side = row["Side"]
    if side == "BUY":
        return mid - spread
    elif side == "SELL":
        return mid + spread
    else:
        raise ValueError(f"Unknown side {side}")
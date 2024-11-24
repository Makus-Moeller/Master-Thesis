import numpy as np
import pandas as pd
from utils import * 
from DeepRL.datatransformer import DataTransformerDeep
from algortihms import *
from typing import Type
import xgboost as xgb


class ContextualRFQAgent():
    def __init__(
        self,
        data_transformer: Type['DataTransformerDeep'] 
        
    ):
        self._data_transformer = data_transformer
        self._model = None
        self._label_mapper = None

    def predict(self, data, algorithm='action_classifier', extended_dataset=True, calibrated=False):
        assert self._model is not None, "Model is not trained"
        #Transform validation data
        df, original_indexes = self._data_transformer.transform_dataset(data, apply_train_bins=True)
       
        #get actions
        if algorithm == 'action_classifier':
            feature_values = pd.DataFrame(df['State'].tolist())
            labels = self._model.predict(feature_values)
            actions = [self._label_mapper[label] for label in labels]
            spreads = [map_int_to_spread(x, self._data_transformer.lower_spread_limit, self._data_transformer.upper_spread_limit) for x in actions]
        
        elif algorithm == 'reward_regression': #Also called CRM
            feature_values = pd.DataFrame(df['State'].tolist())
            #print(feature_values)
            num_actions = int((self._data_transformer.upper_spread_limit - self._data_transformer.lower_spread_limit)*1000)
            possible_actions = np.arange(num_actions)  # Actions range from 0 to 154

            # Define a function to find the best action for each trade context
            def get_best_action_for_each_trade(df, model, possible_actions):
                best_actions = []  # To store the best action for each row

                for _, row in df.iterrows():
                    context_data = pd.DataFrame([row] * len(possible_actions))  # Duplicate the row for each action
                    context_data['Action'] = possible_actions  # Assign each possible action

                    predicted_rewards = model.predict(context_data)

                    best_action = possible_actions[np.argmax(predicted_rewards)]
                    best_actions.append(best_action)

                return best_actions

            # Usage:
            # best_actions will contain the best action for each trade (each row in df)
            best_actions = get_best_action_for_each_trade(feature_values, self._model, possible_actions)
            
            spreads = [map_int_to_spread(x, self._data_transformer.lower_spread_limit, self._data_transformer.upper_spread_limit) for x in best_actions]
        
        else: raise ValueError(f"Unsupported algorithm: {algorithm}")  

        #Transform to spread and add to mid price
        data = data.loc[original_indexes].copy()
        data["PolicySpread"] = spreads
        data["ModelPrice"] = data.apply(_model_price_calibrated if calibrated else _model_price, axis=1).copy()
        return np.asarray(data["ModelPrice"], np.float64), data if extended_dataset else np.asarray(data["ModelPrice"], np.float64)  

        

    def train(self, dataset: pd.DataFrame, algorithm='action_classifier'):
        df, _ = self._data_transformer.transform_dataset(dataset)
        feature_values = pd.DataFrame(df['State'].tolist())
        
        #Create mappings between spread values and contiguous integer labels. Not all actions are represented in train data
        unique_spreads = sorted(df['Action'].unique())  # This should be target action and not Action
        spread_to_label = {spread: i for i, spread in enumerate(unique_spreads)}  # Spread to label
        label_to_spread = {i: spread for i, spread in enumerate(unique_spreads)}  # Label to spread

        # Step 2: Map target spreads to integer labels
        df['classifier_label'] = df['Action'].map(spread_to_label)  # Add new column with mapped labels
        
        if algorithm == 'action_classifier':
            xgb_model = xgb.XGBClassifier(
                objective="multi:softmax",          # Since target spread is discrete
                num_class=len(unique_spreads),      # Number of unique target classes
                max_depth=6,                
                eta=0.3,                    
                n_estimators=100,           
                random_state=42
                )

            xgb_model.fit(feature_values, df["classifier_label"])
            self._model = xgb_model
            self._label_mapper = label_to_spread
        
        elif algorithm == 'reward_regression':
            context_features = pd.DataFrame(df['State'].tolist())  # Split context features into separate columns
            actions = df['Action']  # Each row's action
            rewards = df['Reward']  # Each row's reward

            # Combine the context features with actions to create input features for CRM
            crm_data = pd.concat([context_features, actions], axis=1)
            xgb_model = xgb.XGBRegressor(
                max_depth=6,
                eta=0.3,
                n_estimators=100,
                random_state=42
            )
            xgb_model.fit(crm_data, rewards)
            self._model = xgb_model 
            
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
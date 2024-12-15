#import numpy as np
import sys
import pandas as pd
sys.path.insert(0, 'C:/Udvikler/Speciale/SpecialeKode')
from utils import * 
#from algortihms import *
#from typing import Type
from typing import Callable
import matplotlib.pyplot as plt 


class DataTransformerDeep():
    def __init__(
        self,
        input_features: list[str],
        lower_spread_limit: float,
        upper_spread_limit: float,
        reward_function: Callable[[pd.DataFrame], pd.DataFrame],
        opponent_actions: bool,
        discretize_actions : bool,
        reward_terms: bool
        ):
        self.input_features = input_features
        self.lower_spread_limit = lower_spread_limit
        self.upper_spread_limit = upper_spread_limit
        self.reward_function = reward_function
        self.opponent_actions = opponent_actions
        self._discretize_actions = discretize_actions
        self.reward_terms = reward_terms
        self._input_features_dummy_extended = None
        self._dummy_variables = None
        self._numeric_columns = None
        self._categorical_columns = None 
        self._FirmAccount_index = None
        self._Side_index = None
        self._BookName_index = None
        self._min_vals = None
        self._max_vals = None


    def transform_dataset(self, dataset, apply_train_bins = False):
        df = dataset.copy()
        df['OriginalIndex'] = df.index
        df = self.add_direction_column(df)
        df = self.add_price_movement(df)
        df = self.add_hitrate_delta(df, 100)
        df = self.add_spread_column(df, self.opponent_actions)
        df = self.add_reward_column(df, reward_function=self.reward_function, reward_terms=self.reward_terms)
        df = self.normalize_reward(df, robust=True)
        #df = self.log_transform_reward(df)
        df = self.filter_features_of_interest(df, self.opponent_actions, self.reward_terms)
        
        #encode categorical features or apply already defined indexes
        if apply_train_bins:
            df = self.encode_categorical_features_apply_records(df)
            df = self.normalize_state_features_apply_limits(df, self._min_vals, self._max_vals)
        else: 
            df = self.encode_categorical_features(df)
            df = self.normalize_state_features(df)
        
        df = self.create_state_vector(df)
        
        df_RL_format = self.create_final_df(df, self._discretize_actions,  
                                            self.opponent_actions,
                                            self.reward_terms)

        df_RL_format.reset_index(inplace=True, drop=True)
        
        if not apply_train_bins:
            df_RL_format.to_csv('../output/RLDeep_df.csv')

        return df_RL_format, df.OriginalIndex


    def add_direction_column(self, df):
        df["Direction"] = df["Side"].apply(lambda x: -1 if x == "BUY" else 1)
        df["Direction"] = df["Direction"].astype(float)
        return df
    
    def add_price_movement(self, df):
        df_sorted = df.sort_values(by=['Isin', 'TradeTime'])
        df_sorted['Price_Diff'] = df_sorted.groupby('Isin')['Mid'].diff().fillna(0)

        # Merge the Price_Diff back to the original DataFrame in time-sorted order
        df = df.merge(df_sorted[['Isin', 'TradeTime', 'Price_Diff']], on=['Isin', 'TradeTime'])
        return df 
    
    def add_hitrate_delta(self, df, window_size):
        df['accepted'] = df['Result'].map({'Won': 1, 'Lost': 0})
        df['h_t'] = df['accepted'].rolling(window=window_size, min_periods=1).mean()
        df.loc[:9, 'h_t'] = 0.3  # Default value for the first 10 rows

        # Compute rolling average of DiC
        df['rolling_avg_thickness'] = df['DealersInCompetition'].rolling(window=window_size, min_periods=1).mean()
        df.loc[:9, 'rolling_avg_thickness'] = df.loc[:9, 'DealersInCompetition'].mean()  # Default average

        # Calculate h_desired_t = 1 / (rolling_avg_thickness + 1) 
        df['h_desired_t'] = 1 / (df['rolling_avg_thickness'] + 1)
        df.loc[:9, 'h_desired_t'] = 0.3  

        # Compute delta_h_t
        df['delta_h_t'] = df['h_desired_t'] - df['h_t']

        return df 

    def add_spread_column(self, df, opponent_actions):
        # Create spread independent of the side (using "Direction"). I could use ALLQ average as mid? 
        df["Spread"] = df["Direction"] * (df["Price"]-df["Mid"])

        if opponent_actions:
            # Calculate Result conditions
            df["ResultWon"] = (df.Result == "Won") | (df.Result == 1)
            df["ResultLost"] = (df.Result == "Lost") | (df.Result == 0)
            
            df.loc[df.ResultWon, "TargetPrice"] = df.CoverPrice
            df.loc[df.ResultLost, "TargetPrice"] = df.TradedPrice

            df["Opponent_Spread"] = df["Direction"] * (df["TargetPrice"]-df["Mid"])
            df = df[(df.Opponent_Spread > self.lower_spread_limit) & (df.Opponent_Spread < self.upper_spread_limit)]
            
        #We want a reasonable number of actions so outliers are removed
        df = df[(df.Spread > self.lower_spread_limit) & (df.Spread < self.upper_spread_limit)]
        
        return df
    
    def add_reward_column(self, df, reward_function, reward_terms):
        return reward_function(df, reward_terms)
    
    
    def normalize(self, v, v_min, v_max):
        if v_max > v_min:  # Prevent division by zero
            return (v - v_min) / (v_max - v_min)
        else:
            return 0.5  

     # Function to normalize a reward to [0, 1]
    def normalize_reward(self, df, robust=False, plot_dist=False):
        if robust:
            low, high = np.percentile(df["Reward"], [5, 95])
            df["Reward_clipped"] = np.clip(df["Reward"], low, high)
            df["Reward"] = (df["Reward_clipped"] - low) / (high - low)
        else:
            # Calculate min and max rewards in the dataset
            r_min = min(df["Reward"])
            r_max = max(df["Reward"])
            df["Reward"] = df["Reward"].map(lambda x: self.normalize(x, r_min, r_max))
        
        #After normalising
        if plot_dist:
            plt.hist(df["Reward"], bins=40 )
        
        return df

    def log_transform_reward(self, df):
        epsilon = 1e-6
        df["Reward"] = np.log1p(df["Reward"] + epsilon)
        plt.hist(df["Reward"], bins=40)
        return df


    def filter_features_of_interest(self, df, opponent_actions, reward_terms):
        columns_of_interest = self.input_features + ["Reward", "Spread", "OriginalIndex"] 
        
        if opponent_actions:
            columns_of_interest.append("Opponent_Spread")
        
        if reward_terms:
            columns_of_interest.append("Reward_Terms")
        
        df = df[columns_of_interest]
        df = df.dropna()
        return df
    
    def encode_categorical_features(self, df):

        #Extract categorical features
        categorical_columns = df.select_dtypes(include=["object"]).columns
        self._categorical_columns = [col for col in categorical_columns if col != "Reward_Terms"]


        # Create mappings for categorical features
        for col in categorical_columns:
           
            # Additional functionality for "FirmAccount"
            if col == "FirmAccount":
                df[col] = df[col].mask(df[col].map(df[col].value_counts(normalize=True)) < 0.03, 'Other')
                
            value_to_index = {value: idx for idx, value in enumerate(df[col].unique())}
            setattr(self, f"_{col}_index", value_to_index)
            #df[col] = df[col].map(value_to_index)

        df_encoded = pd.get_dummies(df, columns=self._categorical_columns, drop_first=False)
        
        # Update input_features with the new columns
        dummy_variables = df_encoded.columns.difference(df.columns)  # Find new columns created by one-hot encoding
        self._dummy_variables = list(dummy_variables)
        self._input_features_dummy_extended = [feat for feat in self.input_features if feat not in self._categorical_columns]  # Remove original categorical columns
        self._input_features_dummy_extended += list(dummy_variables)  # Add new columns to input_features

        return df_encoded
    
    def encode_categorical_features_apply_records(self, df):
        assert self._categorical_columns is not None, "Model has not been trained, categorical columns are missing"
        # Separate categorical and non-categorical features
        categorical_df = df[self._categorical_columns]
        non_categorical_df = df.drop(columns=self._categorical_columns)

        # One-hot encode only the categorical features
        categorical_encoded = pd.get_dummies(categorical_df, columns=self._categorical_columns, drop_first=False)

        # Concatenate non-categorical features with the one-hot encoded categorical features
        df_encoded = pd.concat([non_categorical_df, categorical_encoded], axis=1)

        # Reindex to ensure the same columns are used as in training (self._dummy_variables should be a list of columns used during training)
        df_encoded = df_encoded.reindex(columns=self._input_features_dummy_extended + ["Reward", "Spread", "OriginalIndex", "Reward_Terms"], fill_value=0)  # Fill missing columns with 0

        return df_encoded
        

    def normalize_state_features(self, df):
        mins = []
        maxs = []        
        for feature in self._input_features_dummy_extended:
            v_min = min(df[feature])
            v_max = max(df[feature])
            
            #normalize feature
            df[feature] = df[feature].map(lambda x: self.normalize(x, v_min, v_max))
            
            #save min max value for prediction
            mins.append(v_min)
            maxs.append(v_max)
        
        self._min_vals = mins
        self._max_vals = maxs
        return df
    
    def normalize_state_features_apply_limits(self, df, mins, maxs):        
        for feature, v_min, v_max in zip(self._input_features_dummy_extended, mins, maxs):
            
            #normalize feature
            df[feature] = df[feature].map(lambda x: self.normalize(x, v_min, v_max))
        
        return df
    
    
    
    def create_state_vector(self, df):
        
        # Create a new column 'State' containing lists of values (statevector) from the relevant bin features
        df['State'] = df.apply(lambda row: [row[feat] for feat in self._input_features_dummy_extended], axis=1)
        
        return df


    def create_final_df(self, df, discretize_actions, opponent_actions, reward_terms):
        # Create a new DataFrame with the first column as the lists of combined features
        final_df = pd.DataFrame({'State': df['State'], 'Action': df['Spread'], 'Reward': df['Reward']})

        if discretize_actions:
            final_df["Action"] = df["Spread"].map(lambda x: map_spread_to_int(x, self.lower_spread_limit, self.upper_spread_limit))

        if opponent_actions:
            if discretize_actions:
                final_df["Opponent_Action"] = df["Opponent_Spread"].map(lambda x: map_spread_to_int(x, self.lower_spread_limit, self.upper_spread_limit))
            else:
                final_df["Opponent_Action"] = df["Opponent_Spread"]
        
        if reward_terms:
            final_df["Reward_Terms"] = df["Reward_Terms"]

        # Shift the State column
        final_df["Next_State"] = final_df['State'].shift(-1)
        
        final_df.reset_index(drop=True, inplace=True)
        
        # Create a list of -1 with the same length as the elements in the State column
        fill_value = [-1.0] * len(final_df['State'][0])  # Adjust this to the appropriate length

        # Manually assign the fill value to the last row of Next_State
        final_df["Next_State"].iloc[-1] = fill_value  
        
        return final_df        

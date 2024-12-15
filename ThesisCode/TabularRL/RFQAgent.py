import numpy as np
import pandas as pd
from utils import * 
from algortihms import *
from typing import Type
from typing import Callable
import matplotlib.pyplot as plt 


class RFQAgent():
    def __init__(
        self,
        data_transformer: Type['DataTransformer'] 
        
    ):
        self._data_transformer = data_transformer
        self._policy = None
        self._random_action_generator = None
        self._train_state_indices = None
        

    def predict(self, data, algorithm=None, extended_dataset=True, calibrated=False, random_policy=False):
        assert self._policy is not None or self._random_action_generator is not None, "Model is not trained"
        df, original_indexes = self._data_transformer.transform_dataset(data, apply_train_bins=True)
        unknownStateCoverage(test_df=df, train_state_indices=self._train_state_indices)

        state_indices = df["State_index"].astype(int).values
        
        #Get actions
        if algorithm in ['random_by_action_distr', 'random_by_uniform']:
            actions = np.array([self._random_action_generator() for _ in range(len(df))])
        else: 
            actions = self._policy[state_indices]
        
        #Map to spread and finally prices
        spreads = [map_int_to_spread(x, self._data_transformer.lower_spread_limit, self._data_transformer.upper_spread_limit) for x in actions]
        #add spread and price to original data
        data = data.loc[original_indexes].copy()
        data["PolicySpread"] = spreads
        data["ModelPrice"] = data.apply(_model_price_calibrated if calibrated else _model_price, axis=1)
        
        return np.asarray(data["ModelPrice"], np.float64), data if extended_dataset else np.asarray(data["ModelPrice"], np.float64)  

    def train(self, dataset: pd.DataFrame, algorithm='QL', gamma=0.99):
        df, _ = self._data_transformer.transform_dataset(dataset)
        self._train_state_indices = stateCoverageInformation(df, self._data_transformer)
        
        num_actions = int((self._data_transformer.upper_spread_limit - self._data_transformer.lower_spread_limit)*1000)
        num_states = np.prod(self._data_transformer._num_values)
            
        if algorithm=='QL':
            optimal_policy_QL = QL(df, gamma, num_states, num_actions)
            self._policy = optimal_policy_QL
        
        elif algorithm=='DQL':
            optimal_policy_DQL = double_QL(df, gamma, num_states, num_actions)
            self._policy = optimal_policy_DQL
        
        elif algorithm=='QL_MarkovGame':
            optimal_policy_QL = QL_MarkovGame(df, gamma, num_states, num_actions)
            self._policy = optimal_policy_QL
        
        elif algorithm=='double_QL_MarkovGame':
            optimal_policy_DQL = double_QL_MarkovGame(df, gamma, num_states, num_actions)
            self._policy = optimal_policy_DQL
        
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



class DataTransformer():
    def __init__(
        self,
        input_features: list[str],
        lower_spread_limit: float,
        upper_spread_limit: float,
        num_bins: int,
        discretize_method: str,
        reward_function: Callable[[pd.DataFrame], pd.DataFrame],
        opponent_actions: bool,
        vectorize_next_state: bool,
        reward_terms: bool
        ):
        self.input_features = input_features
        self.lower_spread_limit = lower_spread_limit
        self.upper_spread_limit = upper_spread_limit
        self.num_bins = num_bins
        self.discretize_method = discretize_method 
        self.reward_function = reward_function
        self.opponent_actions = opponent_actions
        self.vectorize_next_state = vectorize_next_state
        self.reward_terms = reward_terms
        self._numeric_columns = None
        self._categorical_columns = None 
        self._num_values = None #list of cardinality size for each discretized feature
        self._train_bin_edges = None
        self._train_quantiles = None 
        self._FirmAccount_index = None
        self._Side_index = None
        self._BookName_index = None


    def transform_dataset(self, dataset, apply_train_bins = False):
        df = dataset.copy()
        df['OriginalIndex'] = df.index
        df = self.add_direction_column(df)
        df = self.add_price_movement(df)
        df = self.add_spread_column(df, self.opponent_actions)
        df = self.add_reward_column(df, reward_function=self.reward_function, reward_terms=self.reward_terms)
        df = self.normalize_reward(df, robust=True, plot_dist=False)
        df = self.filter_features_of_interest(df, self.opponent_actions, self.reward_terms)
        
        #Make discretization in train or apply already defined bin edges/quantiles
        if apply_train_bins:
            df = self.discretize_features_apply_records(df)
        else: 
            df = self.discretize_features(df)
        
        
        df = self.create_state_vector(df)
        df_RL_format = self.create_state_and_action_mapping(df, self.opponent_actions, 
                                                            vectorize_next_state=self.vectorize_next_state, 
                                                            reward_terms=self.reward_terms)

        df_RL_format.reset_index(inplace=True, drop=True)

        if not apply_train_bins:
            df_RL_format.to_csv('output/RL_df.csv')

        return df_RL_format, df.OriginalIndex


    def add_direction_column(self, df):
        df["Direction"] = df["Side"].apply(lambda x: -1 if x == "BUY" else 1)
        df["Direction"] = df["Direction"].astype(float)
        return df
    
    def add_price_movement(self, df):
        df_sorted = df.sort_values(by=['Isin', 'TradeTime'])
        
        # Calculate the price difference for each ISIN and set NaN values to 0
        df_sorted['Price_Diff'] = df_sorted.groupby('Isin')['Mid'].diff().fillna(0)

        # Merge the Price_Diff back to the original DataFrame in time-sorted order
        df = df.merge(df_sorted[['Isin', 'TradeTime', 'Price_Diff']], on=['Isin', 'TradeTime'])
        return df 

    def add_spread_column(self, df, Opponent_actions):
        # Create spread independent of the side (using "Direction"). I could use ALLQ average as mid? 
        df["Spread"] = df["Direction"] * (df["Price"]-df["Mid"])

        if Opponent_actions:
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
                return 0.5  # In case all rewards are the same, map them to 0.5

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

    def filter_features_of_interest(self, df, Opponent_actions, reward_terms):
        #Features of interest
        columns_of_interest = self.input_features + ["Reward", "Spread", "OriginalIndex"] 
        
        if Opponent_actions:
            columns_of_interest.append("Opponent_Spread")

        if reward_terms:
            columns_of_interest.append("Reward_Terms")
        
        df = df[columns_of_interest]
        df = df.dropna()
        return df
    
    def discretize_features(self, df):
        #Extract numerical features of interest
        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
        columns_to_exclude = ['Spread', 'Opponent_Spread', 'Reward', 'OriginalIndex', 'Reward_Terms']
        numeric_columns = [col for col in numeric_columns if col not in columns_to_exclude]
        self._numeric_columns = numeric_columns

        #Extract categorical features
        categorical_columns = df.select_dtypes(include=["object"]).columns
        categorical_columns = [col for col in categorical_columns if col not in columns_to_exclude]
        self._categorical_columns = categorical_columns

        # Discretization parameters, two discretize methods. Equal widt or equal number of observations.
        train_bin_edges = {}
        train_quantiles = {}
        # Iterate over all columns (features) in the dataframe
        for col in numeric_columns:
            if self.discretize_method == "width":
                #Equal width
                data_min, data_max = np.min(df[col]), np.max(df[col])
                bin_edges = np.linspace(data_min, data_max, self.num_bins + 1)
                train_bin_edges[col] = bin_edges
                df[f"{col}_bins"] = np.digitize(df[col], bin_edges[1:-1])
            else:
                #Equal number 
                df[f"{col}_bins"], quantiles = pd.qcut(df[col], self.num_bins, labels=False, duplicates="drop", retbins=True)
                train_quantiles[col] = quantiles

        self._train_bin_edges = train_bin_edges
        self._train_quantiles = train_quantiles
        

        # Create mappings for categorical features
        categorical_nums = []
        for col in categorical_columns:
            # Additional functionality for "FirmAccount"
            if col == "FirmAccount":
                df[col] = df[col].mask(df[col].map(df[col].value_counts(normalize=True)) < 0.03, 'Other')
                
            value_to_index = {value: idx for idx, value in enumerate(df[col].unique())}
            setattr(self, f"_{col}_index", value_to_index)
            df[f"{col}_bins"] = df[col].map(value_to_index)
            categorical_nums.append(len(value_to_index))
                
        num_values = [self.num_bins] * len(self._numeric_columns) + categorical_nums
        self._num_values = num_values
        
        return df
    
    def discretize_features_apply_records(self, df):
        assert self._numeric_columns is not None, "Model has not been trained, numeric Columns is missing"
        assert self._categorical_columns is not None, "Model has not been trained, categorical Columns is missing"
        
        
        for col in self._numeric_columns:
            if self.discretize_method == "width":
                #equal width
                bin_edges = self._train_bin_edges.get(col)
                if bin_edges is not None:
                    df[f"{col}_bins"] = np.digitize(df[col], bin_edges[1:-1])
            else:
                #Equal number or frequency in each bin
                quantiles = self._train_quantiles.get(col)
                if quantiles is not None:
                    # Use the 'quantiles' from the training set to discretize the test set
                    df[f"{col}_bins"] = pd.cut(df[col], bins=quantiles, labels=False, duplicates="drop", include_lowest=True)
            
        
        # Create mappings for categorical features
        for col in self._categorical_columns:
            # Additional functionality for "FirmAccount"
            if col == "FirmAccount":
                value_for_other = self._FirmAccount_index.get('Other')
                df[f'{col}_bins'] = df[col].map(self._FirmAccount_index).fillna(value_for_other)
            else:
                df[f"{col}_bins"] = df[col].map(getattr(self, f"_{col}_index"))
                
        # pd.cut can retun NaN if value test set is outside bins. Thus I have to remove those 
        df = df.dropna()

        return df
    

    def create_state_vector(self, df):
        #Final bin features
        bin_features = [f"{item}_bins" for item in self._numeric_columns]
        bin_features += [f"{item}_bins" for item in self._categorical_columns]

        # Create a new column 'State' containing lists of values (statevector) from the relevant bin features
        df['State'] = df.apply(lambda row: [row[feat] for feat in bin_features], axis=1)
        
        return df

    def create_state_and_action_mapping(self, df, Opponent_actions, vectorize_next_state, reward_terms):
        # Create a new DataFrame with the first column as the lists of combined features
        final_df = pd.DataFrame({'State': df['State']})

        final_df["State_index"] = final_df['State'].map(lambda x: state_to_index(x, self._num_values)) 
        final_df["Action"] = df["Spread"].map(lambda x: map_spread_to_int(x, self.lower_spread_limit, self.upper_spread_limit))
        final_df["Reward"] = df["Reward"]
        
        if Opponent_actions:
            final_df["Opponent_Action"] = df["Opponent_Spread"].map(lambda x: map_spread_to_int(x, self.lower_spread_limit, self.upper_spread_limit))
        
        if reward_terms:
            final_df["Reward_Terms"] = df["Reward_Terms"]

        #Set last next_state to -1 to indicate invalid type
        if vectorize_next_state:
            # Shift the State column
            final_df["Next_State"] = final_df['State'].shift(-1)
            
            final_df.reset_index(drop=True, inplace=True)
            
            # Create a list of -1 with the same length as the elements in the State column
            fill_value = [-1] * len(final_df['State'][0])  # Adjust this to the appropriate length

            # Manually assign the fill value to the last row of Next_State
            final_df["Next_State"].iloc[-1] = fill_value  # Set the last entry to the fill value
        
        else:
            final_df["Next_State"] = final_df['State_index'].shift(-1).fillna(-1).astype(int)

        return final_df        



    
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt 

#Maybe let first real action be 1, beacuse Q-table is initialized with zeros, so a bit confusing that the policy outputtet gives zero for most states

def map_spread_to_int(value, min_value, max_value):
    """
    Maps a value within a specified range to an integer index.

    Args:
        value (float): The value to be mapped.
        min_value (float): The minimum value of the range.
        max_value (float): The maximum value of the range.
        num_indices (int): The total number of indices.

    Returns:
        int: The mapped integer index.
    """
    num_indices = (max_value - min_value)*1000
    
    # Normalize the value within the range to [0, 1]
    normalized_value = (value - min_value) / (max_value - min_value)

    # Multiply by the total number of indices
    scaled_value = normalized_value * num_indices
    mapped_index = int(scaled_value)

    return mapped_index


def map_int_to_spread(mapped_index, min_value, max_value):
    """
    Maps an integer index to a value within a specified range.

    Args:
        mapped_index (int): The integer index to be mapped.
        min_value (float): The minimum value of the range.
        max_value (float): The maximum value of the range.

    Returns:
        float: The mapped value.
    """
    num_indices = (max_value - min_value) * 1000

    # Calculate the normalized value from the integer index
    normalized_value = mapped_index / num_indices

    # Scale back to the original range
    value = min_value + normalized_value * (max_value - min_value)
    rounded_value = round(value, 3)
    
    return rounded_value

def map_int_to_spread_tensor(mapped_index_tensor, min_value, max_value):
    """
    Maps each integer index in a tensor to a value within a specified range.

    Args:
        mapped_index_tensor (Tensor): A tensor of integer indices to be mapped.
        min_value (float): The minimum value of the range.
        max_value (float): The maximum value of the range.

    Returns:
        Tensor: A tensor of mapped values within the specified range.
    """
    num_indices = (max_value - min_value) * 1000

    # Calculate normalized values for each integer index in the tensor
    normalized_values = mapped_index_tensor.float() / num_indices

    # Scale back to the original range
    values = min_value + normalized_values * (max_value - min_value)

    # Round each element in the tensor to 3 decimal places
    rounded_values = torch.round(values * 1000) / 1000

    return rounded_values


def state_to_index(state, num_values):
    '''
    Convert a multi-dimensional state to a single index.

    :param state: A list of feature values representing the state.
    :param num_values: A list of the number of possible values for each feature.
    :return: A single integer representing the index of the state in the Q-table. 
    '''
    
    # Ensure that the state and num_values have the same size
    assert len(state) == len(num_values), "State and num_values must have the same length."
    
    index = 0
    multiplier = 1
    for i in reversed(range(len(state))):
        index += state[i] * multiplier
        multiplier *= num_values[i]
        
    return index

def index_to_state(index, num_values):
    '''
    Convert a single index back to a multi-dimensional state.

    :param index: A single integer representing the index of a state.
    :param num_values: A list of the number of possible values for each feature.
    :return: A list of feature values representing the state. 
    ''' 
    state = []
    for n in reversed(num_values):
        state.append(index % n)
        index //= n
    state.reverse()
    return state


def split_data(data: pd.DataFrame, training_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Sort dataset på Tradetime
    data.sort_values('TradeTime', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Calculate the index representing the trainsize cutoff
    cutoff_index = int(len(data) * training_size)

    # Split the dataset into train and test based on the cutoff index
    train = data[:cutoff_index]
    test = data[cutoff_index:]

    return train, test


##################### Reward functions ###########################################


def calculate_margins(df):
    """
    Helper function to calculate common margin-related metrics.
    """
    dataset = df.copy()
    dataset["Margin"] = 0.0
    dataset["Reward"] = 0.0

    # Calculate Result conditions
    dataset["ResultWon"] = (dataset.Result == "Won") | (dataset.Result == 1)
    dataset["ResultLost"] = (dataset.Result == "Lost") | (dataset.Result == 0)
    
    # Calculate Side conditions
    dataset["SideBuy"] = (dataset.Side == "BUY") | (dataset.Side == 1)
    dataset["SideSell"] = (dataset.Side == "SELL") | (dataset.Side == 0)

    quotedPrice = dataset["Price"]
    mid = dataset["Mid"]

    # Set Target Price based on result
    dataset.loc[dataset.ResultWon, "TargetPrice"] = dataset.CoverPrice
    dataset.loc[dataset.ResultLost, "TargetPrice"] = dataset.TradedPrice
    targetPrice = dataset["TargetPrice"]

    # Calculate Spreads
    dataset.loc[dataset.SideBuy & (targetPrice > 0), "TargetSpread"] = mid - targetPrice
    dataset.loc[dataset.SideSell & (targetPrice > 0), "TargetSpread"] = targetPrice - mid

    # Calculate Margin
    dataset.loc[targetPrice > 0, "Margin"] = np.abs(targetPrice - quotedPrice)
    dataset.loc[targetPrice > 0, "MarginSigned"] = dataset["TargetSpread"] - dataset["Spread"]
    
    return dataset


def phi(z):
    return z**2/2

def linearMarginReward(df, reward_terms, alpha=0.5, w1=1.5, w2=0.05):
    dataset = calculate_margins(df)

    dataset = dataset.dropna(subset=["MarginSigned"])
    rewardshift = 1.0
    

    #Calculate inevntory risk standardized
    mean_inventory = np.mean(dataset.inventoryRisk) 
    std_inventory = np.std(dataset.inventoryRisk)
    
    inventory_won = (dataset.inventoryRisk-dataset.Direction*dataset.Amount)
    inventory_lost = (dataset.inventoryRisk)

    inventory_won_standard = (inventory_won - mean_inventory) / std_inventory
    inventory_lost_standard = (inventory_lost - mean_inventory) / std_inventory
    
    #CLAMP VARIBALES
    inventory_won_standard = inventory_won_standard.clip(lower=0.0, upper=1.0)
    inventory_lost_standard = inventory_lost_standard.clip(lower=0.0, upper=1.0)

    # Calculate Rewards
    dataset.loc[dataset.ResultWon, "Reward"] = w1 * (-alpha * dataset.MarginSigned) - w2 * phi(inventory_won_standard) + rewardshift
    dataset.loc[dataset.ResultLost, "Reward"] = w1 * dataset.MarginSigned - w2 * phi(inventory_lost_standard) + rewardshift

    return dataset[dataset.TargetPrice > 0]


def quadraticMarginReward(df, reward_terms, alpha=0.005, w1=1.5, w2=0.05):
    dataset = calculate_margins(df)
    
    dataset = dataset.dropna(subset=["MarginSigned"])
    rewardshift = 1.0
    
    #Calculate inevntory risk standardized
    mean_inventory = np.mean(dataset.inventoryRisk) 
    std_inventory = np.std(dataset.inventoryRisk)
    
    inventory_won = (dataset.inventoryRisk-dataset.Direction*dataset.Amount)
    inventory_lost = (dataset.inventoryRisk)

    inventory_won_standard = (inventory_won - mean_inventory) / std_inventory
    inventory_lost_standard = (inventory_lost - mean_inventory) / std_inventory
    
    #CLAMP VARIBALES
    inventory_won_standard = inventory_won_standard.clip(lower=0.0, upper=1.0)
    inventory_lost_standard = inventory_lost_standard.clip(lower=0.0, upper=1.0)

    # Calculate Rewards
    dataset.loc[dataset.ResultWon, "Reward"] = w1 * (-dataset.MarginSigned ** 2 + alpha) - w2 * phi(inventory_won_standard) + rewardshift
    dataset.loc[dataset.ResultLost, "Reward"] = w1 * (-dataset.MarginSigned ** 2) - w2 * phi(inventory_lost_standard) + rewardshift

    return dataset[dataset.TargetPrice > 0]


def simpleSpreadReward(df):
    """
    Simple reward function calculated as [ResultWon]*Spread
    """
    dataset = calculate_margins(df)

    # Calculate Rewards
    dataset.loc[dataset.ResultWon, "Reward"] = dataset.Spread   
    dataset.loc[dataset.ResultLost, "Reward"] = 0

    return dataset


def SpreadProfitReward(df):
    """
    Simple reward function calculated as [ResultWon]*Spread*Amount
    """
    dataset = calculate_margins(df)

    # Calculate Rewards
    dataset.loc[dataset.ResultWon, "Reward"] = dataset.Spread * dataset.Amount   
    dataset.loc[dataset.ResultLost, "Reward"] = 0

    return dataset


'''def inventory_PnL_reward(df, reward_terms):
    """
    The reward function often used in the litterature e.g. Avellanda & Stoikov 
    """
    dataset = calculate_margins(df)

    #Normalizing constants
    mean_spread_PnL = np.mean(dataset.Amount*dataset.Spread)
    std_inventory_PnL = np.std(dataset.Position_Risk*dataset.Price_Diff)
    std_inventory_risk = np.std(dataset.inventoryRisk)

    # Spread PnL
    spread_PnL = (dataset.Amount*dataset.Spread)/mean_spread_PnL

    #Inventory PnL
    inventory_PnL = (dataset.Position_Risk * dataset.Price_Diff) / std_inventory_PnL
    
    #Inventory penalty, one for win and one for lost 
    inventory_penalty_won = ((dataset.inventoryRisk-((dataset.Direction*dataset.Amount)/1000))/std_inventory_risk)**2/2 #InventoryRisk is divided by 1000
                                                                                                #BUY is direction -1 so have to use minus to add Amount
    inventory_penalty_lost = (dataset.inventoryRisk/std_inventory_risk)**2/2 

    
    #Final rewards
    dataset.loc[dataset.ResultWon, "Reward"] = spread_PnL + inventory_PnL - inventory_penalty_won 
    dataset.loc[dataset.ResultLost, "Reward"] = inventory_PnL - inventory_penalty_lost
                                                


    # If reward_terms is True, collect specified terms in a list
    if reward_terms:    
        # Collect all terms in a list for each row
        dataset['Reward_Terms'] = list(zip(
            [mean_spread_PnL] * len(dataset),
            dataset['Amount'],
            inventory_PnL,
            dataset['Direction'],
            dataset['inventoryRisk'],
            [std_inventory_risk] * len(dataset)
        ))

    return dataset
'''

def inventory_PnL_reward(df, reward_terms, w1=1.0, w2=1.0, w3=0.5):
    """
    Reward function based on inventory PnL and risk, standardized and modularized.
    """
    dataset = calculate_margins(df)

    # Standardize Spread PnL
    spread_PnL = dataset.Amount * dataset.Spread
    mean_spread_PnL = np.mean(spread_PnL)
    std_spread_PnL = np.std(spread_PnL)
    spread_PnL_standard = (spread_PnL - mean_spread_PnL) / std_spread_PnL

    # Standardize Inventory PnL
    inventory_PnL = dataset.Position_Risk * dataset.Price_Diff
    mean_inventory_PnL = np.mean(inventory_PnL)
    std_inventory_PnL = np.std(inventory_PnL)
    inventory_PnL_standard = (inventory_PnL - mean_inventory_PnL) / std_inventory_PnL

    # Standardize Inventory Risk
    mean_inventory_risk = np.mean(dataset.inventoryRisk)
    std_inventory_risk = np.std(dataset.inventoryRisk)
    inventory_risk_standard = (dataset.inventoryRisk - mean_inventory_risk) / std_inventory_risk

    # Apply clamping to standardized variables
    inventory_risk_standard_clamped = inventory_risk_standard.clip(lower=0.0, upper=1.0)

    # Compute Inventory Penalty
    inventory_penalty_won = (((dataset.inventoryRisk - (dataset.Direction * dataset.Amount / 1000))-mean_inventory_risk)/std_inventory_risk).clip(lower=0.0, upper=1.0)
    inventory_penalty_lost = inventory_risk_standard_clamped

    # Final rewards
    dataset.loc[dataset.ResultWon, "Reward"] = (
        w1 * spread_PnL_standard +
        w2 * inventory_PnL_standard -
        w3 * phi(inventory_penalty_won)
    )
    dataset.loc[dataset.ResultLost, "Reward"] = (
        w2 * inventory_PnL_standard -
        w3 * phi(inventory_penalty_lost)
    )

    # If reward_terms is True, collect specified terms in a list
    if reward_terms:
        dataset['Reward_Terms'] = list(zip(
            [mean_spread_PnL] * len(dataset),
            [std_spread_PnL] * len(dataset),
            dataset['Amount'],
            inventory_PnL_standard,
            dataset['Direction'],
            dataset['inventoryRisk'],
            [std_inventory_risk] * len(dataset),
            [mean_inventory_risk] * len(dataset)
        ))


    return dataset

'''def risk_PnL_reward(df, reward_terms):
    """
    Revised reward function often beacuse we have a measure of risk 
    """
    dataset = calculate_margins(df)
    
    #Normalizing constants
    mean_spread_PnL = np.mean(dataset.Amount*dataset.Spread)
    std_inventory_PnL = np.std(dataset.Position_Risk*dataset.Price_Diff)
    std_bpv_risk = np.std(dataset.bpv_risk)
    
    # Spread PnL
    spread_PnL = (dataset.Amount*dataset.Spread)/mean_spread_PnL

    #Inventory PnL
    inventory_PnL = (dataset.Position_Risk * dataset.Price_Diff) / std_inventory_PnL
    
    
    #Inventory penalty, one for win and one for lost 
    risk_penalty_won = ((dataset.bpv_risk -((dataset.Direction*dataset.Amount*dataset.Bpv)/1000))/std_bpv_risk)**2/2 #InventoryRisk is divided by 1000
                                                                                                #BUY is direction -1 so have to use minus to add Amount
    risk_penalty_lost = (dataset.bpv_risk/std_bpv_risk)**2/2 
  
   # Final rewards
    dataset.loc[dataset.ResultWon, "Reward"] = spread_PnL + inventory_PnL - risk_penalty_won
    
    # Losing result rewards
    dataset.loc[dataset.ResultLost, "Reward"] = inventory_PnL - risk_penalty_lost 
    
     # If reward_terms is True, collect specified terms in a list
    if reward_terms:    
        # Collect all terms in a list for each row
        dataset['Reward_Terms'] = list(zip(
            [mean_spread_PnL] * len(dataset),
            dataset['Amount'],
            inventory_PnL,
            dataset['Direction'],
            dataset['bpv_risk'],
            [std_bpv_risk] * len(dataset),
            dataset["Bpv"]
        ))


    return dataset
'''

def risk_PnL_reward(df, reward_terms, w1=1.0, w2=1.0, w3=0.5):
    """
    Reward function based on inventory PnL and risk, standardized and modularized.
    """
    dataset = calculate_margins(df)

    # Standardize Spread PnL
    spread_PnL = dataset.Amount * dataset.Spread
    mean_spread_PnL = np.mean(spread_PnL)
    std_spread_PnL = np.std(spread_PnL)
    spread_PnL_standard = (spread_PnL - mean_spread_PnL) / std_spread_PnL

    # Standardize Inventory PnL
    inventory_PnL = dataset.Position_Risk * dataset.Price_Diff
    mean_inventory_PnL = np.mean(inventory_PnL)
    std_inventory_PnL = np.std(inventory_PnL)
    inventory_PnL_standard = (inventory_PnL - mean_inventory_PnL) / std_inventory_PnL

    # Standardize Bpv Risk
    mean_bpv_risk = np.mean(dataset.bpv_risk)
    std_bpv_risk = np.std(dataset.bpv_risk)
    bpv_risk_standard = (dataset.bpv_risk - mean_bpv_risk) / std_bpv_risk

    # Apply clamping to standardized variables
    bpv_risk_standard_clamped = bpv_risk_standard.clip(lower=0.0, upper=1.0)

    # Compute Inventory Penalty
    bpvRisk_penalty_won = (((dataset.bpv_risk - ((dataset.Direction * dataset.Amount*dataset.Bpv) / 1000))-mean_bpv_risk)/std_bpv_risk).clip(lower=0.0, upper=1.0)
    bpvRisk_penalty_lost = bpv_risk_standard_clamped

    # Final rewards
    dataset.loc[dataset.ResultWon, "Reward"] = (
        w1 * spread_PnL_standard +
        w2 * inventory_PnL_standard -
        w3 * phi(bpvRisk_penalty_won)
    )
    dataset.loc[dataset.ResultLost, "Reward"] = (
        w2 * inventory_PnL_standard -
        w3 * phi(bpvRisk_penalty_lost)
    )

    # If reward_terms is True, collect specified terms in a list
    if reward_terms:
        dataset['Reward_Terms'] = list(zip(
            [mean_spread_PnL] * len(dataset),
            [std_spread_PnL] * len(dataset),
            dataset['Amount'],
            inventory_PnL_standard,
            dataset['Direction'],
            dataset['bpv_risk'],
            [std_bpv_risk] * len(dataset),
            [mean_bpv_risk] * len(dataset),
            dataset["Bpv"]
        ))

    return dataset




def hitrate_target_reward(df, reward_terms, alpha=0.5, w1=1.0, w2=1.0, w3=1.0):
    """
    Revised reward function with measures of risk.
    """
    # Calculate margins
    dataset = calculate_margins(df)
    dataset = dataset.dropna(subset=["MarginSigned"])
    
    rewardshift = 1.0
    
    # Spread PnL
    #Stanadardized
    spread_PnL = dataset.Amount * dataset.Spread
    mean_spread_PnL = np.mean(spread_PnL)
    std_spread_PnL = np.std(spread_PnL)
    spread_PnL_standard = (spread_PnL - mean_spread_PnL) / std_spread_PnL 
    
    # Margin reward
    
    #standardized
    mean_margin = np.mean(dataset.MarginSigned)
    std_margin = np.std(dataset.MarginSigned)
    
    margin_won = -alpha * dataset.MarginSigned
    margin_won_standard= (margin_won - mean_margin) / std_margin

    margin_lost = dataset.MarginSigned
    margin_lost_standard = (margin_lost - mean_margin) / std_margin
    

    # Calculate hitrate reward
    hitrate_reward = 1 - np.abs(dataset.delta_h_t)
    
    #standardized
    mean_hitrate = np.mean(hitrate_reward)
    std_hitrate = np.std(hitrate_reward)
    hitrate_reward_standard = (hitrate_reward - mean_hitrate)/ std_hitrate

    # Final rewards
    dataset.loc[dataset.ResultWon, "Reward"] = (
        w1 * spread_PnL_standard + w2 * margin_won_standard + w3 * hitrate_reward_standard + rewardshift
    )
    
    # Losing result rewards
    dataset.loc[dataset.ResultLost, "Reward"] = (
        w2 * margin_lost_standard + w3 * hitrate_reward_standard + rewardshift
    )

    return dataset[dataset.TargetPrice > 0]



def stateCoverageInformation(df, data_loader):
    print(f"number of different states in dataset: {len(df['State_index'].value_counts())},")
    print(f"size of statespace: {np.prod(data_loader._num_values)},")
    print(f"state coverage by trajectory: {round(len(df['State_index'].value_counts()) / np.prod(data_loader._num_values), 4)}")
    return set(df["State_index"].unique())
    
# Funtion til at analysere state coverage i testdata
def unknownStateCoverage(test_df, train_state_indices):
    test_state_indices = test_df["State_index"].unique()
    
    # Find ukendte tilstande i testdata, som ikke var i træning
    unseen_states = [s for s in test_state_indices if s not in train_state_indices]
    unseen_count = len(unseen_states)
    total_test_states = len(test_state_indices)
    
    # Udregn andelen af ukendte tilstande i testdataene
    unknown_coverage_ratio = unseen_count / total_test_states if total_test_states > 0 else 0
    
    print(f"Total unique states in test data: {total_test_states}")
    print(f"Number of unseen states in test data: {unseen_count}")
    print(f"Proportion of unseen states: {round(unknown_coverage_ratio, 4)}")
    

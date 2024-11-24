import numpy as np
import pandas as pd
from typing import Union
from DeepRL.DeepRFQAgent import DeepRFQAgent
from ContextualBandits.ContextualRFQAgent import ContextualRFQAgent
from RFQAgent import RFQAgent
from model_evaluation import eval_utilities as eval_utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class RFQModelEval:
    def __init__(self, model: Union[RFQAgent, DeepRFQAgent, ContextualRFQAgent]):
        self.model = model
    
    def eval(self, dataset, algorithm=None, print_post_trade = False, reward_constant=0.005, 
             print_violin_plot=False, calibrated=False, plot_spreads=False, random_policy=False):
        data = dataset.copy()
    
        _, extended_dataset = self.model.predict(data, algorithm=algorithm, calibrated=calibrated, random_policy=random_policy)
        result = self.post_trade_analysis(extended_dataset, reward_constant)
        result = self.add_price_movement(result)
        
        #Add dealer spreads
        result["Direction"] = result["Side"].apply(lambda x: -1 if x == "BUY" else 1)
        result["Direction"] = result["Direction"].astype(float)
        result["Spread"] = result["Direction"] * (result["Price"]-result["Mid"])
        
        if print_violin_plot:
            eval_utils.violin_plot(result, features=['DealerMarginWins', 'MLMarginWins'])
        row = self.parse_results(result) 
        if print_post_trade:
            result.to_csv(f'output/posttrade.csv')
        if plot_spreads:
            print(result["ModelSpread"].value_counts())
            plt.hist(result["ModelSpread"], bins=40)
            plt.hist(result["Spread"], bins=40)
        return row
            

    def add_price_movement(self, df):
        df_sorted = df.sort_values(by=['Isin', 'TradeTime'])
        
        # Calculate the price difference for each ISIN and set NaN values to 0
        df_sorted['Price_Diff'] = df_sorted.groupby('Isin')['Mid'].diff().fillna(0)

        # Merge the Price_Diff back to the original DataFrame in time-sorted order
        df = df.merge(df_sorted[['Isin', 'TradeTime', 'Price_Diff']], on=['Isin', 'TradeTime'])
        return df 

    def post_trade_analysis(self, data, reward_constant):
        dataset = data.copy()
        dataset["DealerProfit"] = 0.0
        dataset["DealerMargin"] = 0.0
        dataset["TargetPrice"] = 0.0
        dataset["MLProfit"] = 0.0
        dataset["MLMargin"] = 0.0

        sideBuy = (dataset.Side == "BUY") | (dataset.Side == 1)
        sideSell = (dataset.Side == "SELL") | (dataset.Side == 0)
        resultWon = (dataset.Result == "Won") | (dataset.Result == 1)
        resultLost = (dataset.Result == "Lost") | (dataset.Result == 0)

        coverPrice = dataset["CoverPrice"]
        dealerPrice = dataset["Price"] 
        modelPrice = dataset["ModelPrice"]
        mid = dataset["Mid"]
        amount = dataset["Amount"]
        dataset.loc[resultWon,"TargetPrice"] = dataset.CoverPrice
        dataset.loc[resultLost,"TargetPrice"] = dataset.TradedPrice
        targetPrice = dataset["TargetPrice"]
        dataset = dataset[dataset.TargetPrice > 0]
        
        dataset.loc[sideBuy, "ModelSpread"] = mid-modelPrice
        dataset.loc[sideSell, "ModelSpread"] = modelPrice-mid
        dataset.loc[sideBuy, "TargetSpread"] = mid-targetPrice
        dataset.loc[sideSell, "TargetSpread"] = targetPrice-mid
        
        # ----------------------Dealer metrics--------------
        dataset.loc[sideBuy & resultWon, "DealerProfit"] = (mid-dealerPrice)*amount
        dataset.loc[sideSell & resultWon, "DealerProfit"] = (dealerPrice-mid)*amount

        dataset.loc[resultWon & (coverPrice > 0), "DealerMarginWins"] = np.abs(coverPrice-dealerPrice)

        dataset.loc[(targetPrice > 0), "DealerMargin"] = np.abs(targetPrice-dealerPrice)
        
         # ------------------ML metrics---------------------
        #spread profit
        dataset.loc[sideBuy & (targetPrice > 0) & (modelPrice >= targetPrice), "MLProfit"] = (mid-modelPrice)*amount
        dataset.loc[sideSell & (targetPrice > 0) & (modelPrice <= targetPrice), "MLProfit"] = (modelPrice-mid)*amount
        
        #cover margin when winning
        dataset.loc[sideBuy & (targetPrice > 0) & (modelPrice >= targetPrice), "MLMarginWins"] = np.abs(targetPrice - modelPrice)
        dataset.loc[sideSell & (targetPrice > 0) & (modelPrice <= targetPrice), "MLMarginWins"] = np.abs(targetPrice - modelPrice)

        #cover margin
        dataset.loc[(targetPrice > 0), "MLMargin"] = np.abs(targetPrice - modelPrice)
        dataset.loc[(targetPrice > 0), "MLMarginSigned"] = dataset['TargetSpread'] - dataset['ModelSpread']

        print(f"Desired HR: {100*(1/(1+np.mean(dataset.DealersInCompetition)))} \n")

        reward_function = self.model._data_transformer.reward_function.__name__

        if reward_function == 'simpleSpreadReward': 
            dataset["RLReward"] = 0.0
            dataset.loc[sideBuy & (targetPrice > 0) & (modelPrice >= targetPrice), "RLReward"] = dataset.ModelSpread   
            dataset.loc[sideSell & (targetPrice > 0) & (modelPrice <= targetPrice), "RLReward"] = dataset.ModelSpread  
        elif reward_function == 'quadraticMArginReward':
            dataset[ "RLReward"] = -dataset.MLMarginSigned**2
            dataset.loc[sideBuy & (targetPrice > 0) & (modelPrice >= targetPrice), "RLReward"] = -dataset.MLMarginSigned**2 + reward_constant
            dataset.loc[sideSell & (targetPrice > 0) & (modelPrice <= targetPrice), "RLReward"] = -dataset.MLMarginSigned**2 + reward_constant
        else:
            dataset[ "RLReward"] = dataset.MLMarginSigned + 1.0
            dataset.loc[sideBuy & (targetPrice > 0) & (modelPrice >= targetPrice), "RLReward"] = -reward_constant * dataset.MLMarginSigned + 1.0
            dataset.loc[sideSell & (targetPrice > 0) & (modelPrice <= targetPrice), "RLReward"] = -reward_constant * dataset.MLMarginSigned + 1.0
        return dataset
    
    
    def parse_results(self, results):
        row = pd.Series()
        
        #aggregated metrics
        row["numtrades"] = eval_utils.numTrades(results)
        
        row["meanDiC"] = eval_utils.meanDiC(results)

        row["dealerHitRate"] = eval_utils.dealerHitrate(results)

        row["mlHitRate"] = eval_utils.MLHitrate(results)

        row["MLWinShare"] = eval_utils.MLWinShare(results)
        
        row["mlDistToCoverMean"] = eval_utils.meanMLDistToCover(results)

        row["mlDistToCoverStd"] = eval_utils.stdMLDistToCover(results)
        
        row["dealerDistToCoverMean"] = eval_utils.meanDealerDistToCover(results)

        row["meanMLProfit"] = eval_utils.meanMLProfit(results)

        row["meanDealerProfit"] = eval_utils.meanDealerProfit(results)

        row["meanMLPnLReward"] = eval_utils.meanMLPnLReward(results)

        row["meanDealerPnLReward"] = eval_utils.meanDealerPnLReward(results)

        row["meanMLSharpeRatio"] = eval_utils.meanMLSharpeRatio(results)

        row["meanDealerSharpeRatio"] = eval_utils.meanDealerSharpeRatio(results)

        row["meanDealerSharpeRatioNoNeg"] = eval_utils.meanDealerSharpeRatioNoNeg(results)

        row["meanRLReward"] = eval_utils.meanRewardML(results)

        row["dealerD2TargetMean"] = eval_utils.meanDealerDistToTarget(results)

        row["MLD2TargetMean"] = eval_utils.meanMLDistTotarget(results)

        row["MLD2TargetStd"] = eval_utils.stdMLDistToTarget(results)

        row["DealerNetInventoryDiff"] = eval_utils.WonAmount(results, "Dealer")

        row["MLNetInventoryDiff"] = eval_utils.WonAmount(results, "ML")
         
        return row


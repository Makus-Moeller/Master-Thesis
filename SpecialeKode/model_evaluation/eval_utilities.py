import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from scipy.spatial import ConvexHull

def numTrades(df):
    return len(df) if len(df) > 0 else 0

def numTradesWithCover(df):
    return len(df[df.CoverPrice > 0]) if len(df) > 0 else 0

def meanDiC(df):
    return np.nanmean(df['DealersInCompetition'])

def MLHitrate(df):
    hr = len(df[df["MLMarginWins"] >= 0]) / len(df[~df["TargetPrice"].isnull()]) * 100 if len(df) > 0 else 0
    return round(hr, 2)

def dealerHitrate(df):
    hr = len(df[df["Result"] == "Won"]) / len(df) * 100 if len(df) > 0 else 0
    return round(hr, 2)
#Profit metrics

def MLProfit(df):
    return np.sum(df["MLProfit"]) if len(df) > 0 else 0

def dealerProfit(df): 
    return np.sum(df["DealerProfit"]) if len(df) > 0 else 0

def meanMLProfit(df): 
    PnL = np.sum(df["MLProfit"]) / len(df[~df["MLProfit"].isnull()]) if len(df) > 0 else 0
    return round(PnL/1000, 1)

def meanDealerProfit(df): 
    PnL = np.sum(df["DealerProfit"]) / len(df[~df["DealerProfit"].isnull()]) if len(df) > 0 else 0
    return round(PnL/1000, 1)
 
def meanDealerProfitNoNeg(df): 
    PnL = np.sum(df[df["DealerProfit"] > 0]["DealerProfit"]) / len(df) if len(df) > 0 else 0
    return round(PnL/1000, 1) 

def meanMLProfitWins(df):
    return np.sum(df["MLProfit"]) / len(df[df["MLMarginWins"] >= 0]) if len(df) > 0 else 0

def meanDealerProfitWins(df): 
    return np.sum(df["DealerProfit"]) / len(df[df["Result"]  == "Won"]) if len(df) > 0 else 0

def meanMLPnLReward(df):
    return np.sum(df["MLProfit"]+df["Position_Risk"]*df["Price_Diff"]) / len(df) if len(df) > 0 else 0 

def meanDealerPnLReward(df):
    return np.sum(df["DealerProfit"]+df["Position_Risk"]*df["Price_Diff"]) / len(df) if len(df) > 0 else 0 

def meanMLKronerReward(df):
    return np.sum(df["MLProfit"] + df["Price_Diff"] * df["Position_Risk"] ) / len(df) if len(df) > 0 else 0 

def meanDealerKronerReward(df):
    return np.sum(df["DealerProfit"] + df["Price_Diff"] * df["Position_Risk"] ) / len(df) if len(df) > 0 else 0 


def meanMLSharpeRatio(df, inventoryPnL=False):
    L = meanMLKronerReward(df) if inventoryPnL else meanMLProfit(df)
    squared_diffs = (df["MLProfit"] + df["Position_Risk"] * df["Price_Diff"] - L)**2 if inventoryPnL else (df["MLProfit"]-L)**2
    total_squared_diffs = np.sum(squared_diffs)  
    sr = L / np.sqrt(total_squared_diffs / len(df)) 
    return round(sr*1000, 2)

def meanDealerSharpeRatio(df, inventoryPnL=False):
    L = meanDealerKronerReward(df) if inventoryPnL else meanDealerProfit(df)
    squared_diffs = (df["DealerProfit"] + df["Position_Risk"] * df["Price_Diff"] - L)**2 if inventoryPnL else (df["DealerProfit"]-L)**2
    total_squared_diffs = np.sum(squared_diffs)  
    sr = L / np.sqrt(total_squared_diffs / len(df)) 
    return round(sr*1000, 2)


def meanDealerSharpeRatioNoNeg(df):
    L = meanDealerProfitNoNeg(df)
    squared_diffs = (df["DealerProfit"].clip(lower=0)-L)**2
    total_squared_diffs = np.sum(squared_diffs)  
    sr = L / np.sqrt(total_squared_diffs / len(df)) 
    return round(sr*1000, 2)



# Distance to cover

def meanMLDistToCover(df):
    d2c = np.nanmean(df[df["MLMarginWins"]>= 0]["MLMarginWins"]) * 100 if len(df) > 0 else 0
    return round(d2c, 2)

def stdMLDistToCover(df):
    return np.std(df[df["MLMarginWins"]>= 0]["MLMarginWins"]) * 100 if len(df) > 0 else 0

def meanDealerDistToCover(df):
    d2c = np.nanmean(df[df["Result"] == "Won"]["DealerMarginWins"]) * 100 if len(df) > 0 else 0
    return round(d2c, 2)
    
def medianMLDistToCover(df):
    return np.nanmedian(df[df["MLMarginWins"]>= 0]["MLMarginWins"]) if len(df) > 0 else 0

def medianDealerDistToCover(df):
    return np.nanmedian(df[df["Result"] == "Won"]["DealerMarginWins"]) if len(df) > 0 else 0

# Distance to target 

def meanMLDistTotarget(df):
    d2t = np.nanmean(df["MLMargin"]) * 100 if len(df) > 0 else 0
    return round(d2t, 2)

def stdMLDistToTarget(df):
    return np.std(df["MLMargin"]) * 100  if len(df) > 0 else 0

def meanDealerDistToTarget(df):
    d2t = np.nanmean(df["DealerMargin"]) * 100 if len(df) > 0 else 0
    return round(d2t, 2)

def medianMLDistToTarget(df):
    return np.nanmedian(df["MLMargin"]) if len(df) > 0 else 0

def medianDealerDistToTarget(df):
    return np.nanmedian(df["DealerMargin"]) if len(df) > 0 else 0

# Amount metrics

def amountSummed(df):
    return np.sum(df["Amount"]) if len(df) > 0 else 0

def MLWinShare(df):
    if len(df) > 0:
        return (np.sum(df[df["MLMarginWins"] >= 0]['Amount']) / np.sum(df['Amount'])) * 100 
    else:
        return 0

def dealerWinShare(df):
    if len(df) > 0:
        return (np.sum(df[df["Result"]  == "Won"]['Amount']) / np.sum(df['Amount'])) * 100
    else:
        return 0
    
#Split on side

def hitrateSide(df, side: str, ML_or_dealer: str):
    '''Side is binary "BUY" or "SELL". ML_or_dealer is binary "ML" or "Dealer"'''
    if len(df) > 0:
        if ML_or_dealer == 'ML':
            return len(df[(df.MLMarginWins >= 0) & (df.Side == side)]) / len(df[df.Side == side]) * 100
        else:
            return len(df[(df.Result == "Won") & (df.Side == side)]) / len(df[df.Side == side]) * 100 
    else:
        return 0

def tradesWonSide(df, side: str, ML_or_dealer: str):
    '''Side is binary "BUY" or "SELL". ML_or_dealer is binary "ML" or "Dealer"'''
    if len(df) > 0:   
        if ML_or_dealer == 'ML':
            return len(df[(df.MLMarginWins >= 0) & (df.Side == side)])
        else:
            return len(df[(df.Result == "Won") & (df.Side == side)]) 
    else:
        return 0
    
def nominalWonSide(df, side: str, ML_or_dealer: str):
    '''Side is binary "BUY" or "SELL". ML_or_dealer is binary "ML" or "Dealer"'''
    if len(df) > 0:    
        if ML_or_dealer == 'ML':
            return np.sum(df[(df.MLMarginWins >= 0) & (df.Side == side)]['Amount'])
        else:
            return np.sum(df[(df.Result == "Won") & (df.Side == side)]['Amount']) 
    else:
        return 0


def WonAmount(df, ML_or_dealer: str):
    if len(df) > 0:
        if ML_or_dealer == 'ML':
            sell_won = np.sum(df[(df.MLMarginWins >= 0) & (df.Side == 'SELL')]['Amount'])
            buy_won = np.sum(df[(df.MLMarginWins >= 0) & (df.Side == 'BUY')]['Amount'])
            return sell_won-buy_won
        else:
            sell_won = np.sum(df[(df.Result == "Won") & (df.Side == 'SELL')]['Amount'])
            buy_won = np.sum(df[(df.Result == "Won") & (df.Side == 'BUY')]['Amount'])
            return sell_won - buy_won


#Spread 

def meanMLSpread(df):
    return np.nanmean(abs(df["ModelPrice"]-df["Mid"])) if len(df) > 0 else 0

def meanDealerSpread(df):
    return np.nanmean(abs(df["Price"]-df["Mid"])) if len(df) > 0 else 0

#Outlier metric 

def meanML_D2C_Outliers(df, percentage):
    '''percentage: Share of largest outliers included in average'''
    if len(df) > 0:
        threshold_ML = df[(df["MLMarginWins"] >= 0)]["MLMarginWins"].quantile(1.0-(percentage/100))
        return np.nanmean(df[(df["MLMarginWins"]>= 0) & (df["MLMarginWins"] >= threshold_ML)]["MLMarginWins"])
    else:
        return 0
    
def meanDealer_D2C_Outliers(df, percentage):
    '''percentage: Share of largest outliers included in average'''
    if len(df) > 0:
        threshold_Dealer = df[(df["Result"] == "Won")]["DealerMarginWins"].quantile(1.0-(percentage/100))
        return np.nanmean(df[(df["Result"] == "Won") & (df["DealerMarginWins"] >= threshold_Dealer)]["DealerMarginWins"])
    else:
        return 0

#Reward metrics

def meanRewardML(df):
    return np.nanmean(df["RLReward"]) if len(df) > 0 else 0
 
def meanRewardDealer(df):
    return np.nanmean(df["DealerReward"]) if len(df) > 0 else 0
 


############################################################## VISUALIZATIONS ################################################################

def visualize_trader_vs_ML_metrics(results_df, deepRL = False):


    sns.set_style(style="whitegrid")

    # Filter the DataFrame for only the hit rate metrics
    metric_combinations = [["mlHitRate", "dealerHitRate"],["mlDistToCoverMean", "dealerDistToCoverMean"], 
                           ["MLD2TargetMean", "dealerD2TargetMean"], ["meanMLProfit", "meanDealerProfit"], 
                           ["meanMLSharpeRatio", "meanDealerSharpeRatio"]]

    for metric_combi in metric_combinations:
    # Reset index to turn the MultiIndex into columns for easier plotting
        results_df_reset = results_df.reset_index()

        # Melt the DataFrame to have a long-form DataFrame suitable for seaborn
        id_vars = ["Features", "Reward Function", "Learning Algorithm", "Sample method", "Alpha"] if deepRL else ["Features", "Reward Function", "Learning Algorithm"] 

        plot_df = results_df_reset.melt(
            id_vars=id_vars,
            value_vars=metric_combi,
            var_name="Metric",
            value_name="Value"
        )

        # Create a new column that combines all the parameter values into a single string
        if deepRL:
            plot_df['Combination'] = (plot_df['Features'] + ' | \n' + plot_df['Reward Function'] + ' | \n' + plot_df['Learning Algorithm'] + 
                                ' | \n' + plot_df['Sample method'] + ' | \n' + plot_df['Alpha'])
        else:
            plot_df['Combination'] = (plot_df['Features'] + ' | \n' + plot_df['Reward Function'] + ' | \n' + plot_df['Learning Algorithm']) 
            
        # Set the size of the plot
        plt.figure(figsize=(16, 8))

        # Create the bar plot using seaborn
        sns.barplot(
            data=plot_df,
            x="Combination",  # Use the combined column for the x-axis
            y="Value",
            hue="Metric",  # Different metrics will have different colors
        )

        # Add titles and labels
        plt.title("Performance by Combination of Features, Reward Function, and Learning Algorithm")
        plt.ylabel("Value")
        plt.xlabel("Combination")

        # Rotate x-axis labels if necessary
        plt.xticks(rotation=90)

        # Display the plot
        plt.show()


def violin_plot(df, features):

    # Filter columns of interest
    df_filtered = df[features]

    # Plot a violin plot
    sns.set_style(style="whitegrid")
    sns.violinplot(data=df_filtered)
    plt.ylim(bottom=0)  
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(f'Distribution of {", ".join(features)}')

    # Create 'output' folder if it doesn't exist
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(output_folder, 'violin_plot.png'))
    plt.show()




def convex_hull(df_multiindex, distance_measure="Cover", deepRL=False):
    
    if distance_measure == "Target":
        distance = "MLD2TargetMean"
        distance_std = "MLD2TargetStd"
    
    else: 
        distance = "mlDistToCoverMean"
        distance_std = "mlDistToCoverStd"
    
    # Your data setup
    df = df_multiindex.reset_index()

    if deepRL:
        df['Combination'] = (df['Features'] + ' | \n' + df['Reward Function'] + ' | \n' + df['Learning Algorithm'] + ' | \n' + df['Sample method']
                         + ' | \n' + df['Alpha'])
    else:
        df['Combination'] = (df['Features'] + ' | \n' + df['Reward Function'] + ' | \n' + df['Learning Algorithm'])
    
    colors = cm.Dark2(np.linspace(0, 1, len(df)))

    mean_DiC = 3 # Hardcoded, remeber to insert real average
    benchmark = 1/(mean_DiC+1)*100
    hitrates = []
    winshares = []
    distances = []
    distance_stds = []
    # Iterate using iloc
    for i in range(len(df)):
        hitrates.append(abs(df.iloc[i]["mlHitRate"]-benchmark))
        winshares.append(abs(df.iloc[i]["MLWinShare"]-benchmark))
        distances.append(df.iloc[i][distance])
        distance_stds.append(df.iloc[i][distance_std])

    fig, ax = plt.subplots()
    names = df['Combination'].to_list()

    # Plotting with convex hull
    for i in range(len(hitrates)): 
        pts_list = [[hitrates[i]+0.0002, distances[i]-distance_stds[i]], #To avoid points has same location, e.g when bad performances 
                    [winshares[i]+0.0001, distances[i]-distance_stds[i]], 
                    [hitrates[i], distances[i]+distance_stds[i]], 
                    [winshares[i], distances[i]+distance_stds[i]]]  
        
        pts = np.array(pts_list)
        hull = ConvexHull(pts)
        plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], alpha=.5, color=colors[i], label=names[i])
        ax.plot((hitrates[i] + winshares[i]) / 2, distances[i], marker='.', color=colors[i], markersize=8)

    # Move the legend to the right of the plot
    plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel('Hitrate/winshare distance to benchmark')
    y_label = f'D2{distance_measure} +- 1 std.'
    plt.ylabel(y_label)
    plt.savefig('output/comparisonPlot.pdf', bbox_inches='tight')
    plt.show()

############################################## Train Loss plots ##############################################################

def plot_loss_progress(*loss_lists, labels=None, title="Loss Progress", ylabel="Loss", xlabel="Iterations"):
    """
    Plot the loss progress dynamically based on the number of loss lists provided.

    :param loss_lists: One or more lists of losses (each list corresponds to a different type of loss).
    :param labels: List of labels for the loss plots. If None, default labels will be used.
    :param title: Title of the plot.
    :param ylabel: Label for the Y-axis (default is "Loss").
    :param xlabel: Label for the X-axis (default is "Iterations").
    """
    sns.set_style(style="whitegrid")
    num_losses = len(loss_lists)  # Determine how many loss lists are provided
    if num_losses == 0:
        print("No loss data provided.")
        return
    
    if labels is None:
        labels = [f"Loss {i+1}" for i in range(num_losses)]  # Default labels if none provided
    
    # Create subplots depending on the number of loss lists
    fig, ax = plt.subplots(num_losses, 1, figsize=(8, 5 * num_losses), squeeze=False)
    
    for i, loss in enumerate(loss_lists):
        ax[i][0].plot(loss, label=labels[i])
        ax[i][0].set_title(f'{labels[i]} Progress')
        ax[i][0].set_xlabel(xlabel)
        ax[i][0].set_ylabel(ylabel)
        ax[i][0].legend()

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the overall title
    plt.show()


def plot_q_target_vs_prediction(pred_q_values, target_q_values, iteration):
    """
    Scatter plot comparing predicted Q-values to target Q-values.
    :param pred_q_values: Predicted Q-values by the current Q-network.
    :param target_q_values: Target Q-values based on the Bellman equation.
    :param iteration: The current training iteration (used in plot title).
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(target_q_values, pred_q_values, alpha=0.6)
    plt.plot([min(target_q_values), max(target_q_values)], [min(target_q_values), max(target_q_values)], 'r--')
    plt.xlim(min(min(target_q_values), min(pred_q_values)), max(max(pred_q_values),max(target_q_values)))
    plt.ylim(min(min(target_q_values), min(pred_q_values)), max(max(pred_q_values),max(target_q_values)))
    plt.xlabel('Target Q-values')
    plt.ylabel('Predicted Q-values')
    plt.title(f'Target vs Predicted Q-values at Iteration {iteration}')
    plt.show()

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from itertools import permutations, combinations
from tqdm import tqdm
import os

import argparse


def decide_on_stationarity(series, alpha=0.05, use_kpss=False):
    result_ADF = adfuller(series, autolag='AIC')
    p_value_ADF = result_ADF[1]
    stationarity_decision_ADF = p_value_ADF < alpha, p_value_ADF  # True if stationary
    if use_kpss :
        result_kpss = kpss(series, regression='c', nlags='auto')
        p_value_kpss = result_kpss[1]
        stationarity_decision_kpss = p_value_kpss > alpha, p_value_kpss  # True if stationary
        return (stationarity_decision_ADF and stationarity_decision_kpss) # Stationary only if both tests say so, otherwise differencing is needed
    else :
        return stationarity_decision_ADF

def preprocess_data_for_grangarcausility(df, alpha=0.05, pearson_correlation_threshold=0.95):
    
    print("Applying Pearson Correlation ....")
    print(f"Original number of channels is {df.shape[1]}")
    corr_matrix = df.corr().abs()  # absolute correlation
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns with correlation above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > pearson_correlation_threshold)]

    print(f"Channels to drop based on very high correlation are {to_drop}")
    
    # Drop all the columns that has to much correlation
    df = df.drop(columns=to_drop)
    print(f"Number of channels after filtering is {df.shape[1]}")

    for col in df.columns:
        count_non_stationary = 0
        series = df[col].dropna()
        is_stationary = decide_on_stationarity(series, alpha)
        while not(is_stationary) :
            # Keep differencing till the series is stationary according to the 2 tests
            print("Channel needed differencing")
            count_non_stationary = count_non_stationary + 1
            series = series.diff().dropna()
            is_stationary = decide_on_stationarity(series, alpha)
        df[col] = series # Make sure to update the channel based on any needed differencing
    print(f"{count_non_stationary} channels needed differencing")
    return df # Return updated dataset


def compute_avg_fscore_per_dataset(dataset, lags_to_test, use_differencing=True, p_val_threshold=0.05):
    channels = dataset.columns
    channel_pairs = list(permutations(channels, 2))  # Ordered pairs: i ➝ j

    total_number_pairs = len(channel_pairs)
    print(total_number_pairs)
    if (total_number_pairs == 0) :
        print("Skipping test as too few channels after filtering...")
        return None
    results = {}

    for lag in lags_to_test:
        f_scores = []
        p_test_scores = []

        # First calculate differenced version of the time series data if required
        if (use_differencing) :
            dataset = dataset.diff().dropna()


        number_grangar_pairs = 0
        for (src, tgt) in tqdm(channel_pairs, desc=f"Testing lag {lag}"):
            try:
                # Data must be 2D: [tgt, src]
                pair_data = dataset[[tgt, src]].dropna().values

                # Run Granger causality
                gc_result = grangercausalitytests(pair_data, maxlag=lag, verbose=False)

                # Extract F-statistic for this lag
                f_val = gc_result[lag][0]['ssr_ftest'][0]
                p_val = gc_result[lag][0]['ssr_ftest'][1]
                print(f"Current pairwise f-score between channel {src} and {tgt} is {f_val}")
                if p_val < p_val_threshold :
                    print(f"Current pairwise p-score between channel {src} and {tgt} is {p_val}")
                    number_grangar_pairs = number_grangar_pairs + 1

                f_scores.append(f_val)

            except Exception as e:
                print(f"Error on pair ({src} ➝ {tgt}) at lag {lag}: {e}")

        # Average F-score for this lag
        avg_f = np.mean(f_scores) if f_scores else np.nan
        results[lag] = avg_f

        # Percentage of pairwise channels that granger causes each other
        percentage = (number_grangar_pairs/total_number_pairs) * 100.0
        print(f"{number_grangar_pairs} Channels out of {total_number_pairs} have grangar causality relation with percentage {percentage}")

    return results

def Test_dataset_causility(dataset, lags_to_test, alpha=0.05) :
    # Preprocess data based on stationary information
    stationary_dataset = preprocess_data_for_grangarcausility(dataset, alpha=alpha)

    # Apply granager causility on the stationary version of the data
    granger_results = compute_avg_fscore_per_dataset(stationary_dataset, lags_to_test)

    return granger_results

parser = argparse.ArgumentParser(description='Arguments for granger causality script')

parser.add_argument('--dataset', type=str, required=True, default="ETTh1.csv", help='Passing here a valid dataset name')

configs = parser.parse_args()

# Based on this we take the first 1000 time steps as the window where the test is applied
time_window = 1000


# Define here the lags
lags = [7, 30, 96, 192]
# Example for running on a specific dataset with many channels


print(configs.dataset)

# Load ETTh1 dataset (replace here with )
dataset = pd.read_csv(f"dataset/{configs.dataset}").iloc[:time_window,1:] # Dropping first column based on 

# Sampling the first 20 channels if the dataset has high dimensionality
if dataset.shape[1] > 21 :
    dataset = dataset[:time_window, 1:21]


# # Measure Grangar Causality of the datasets
grager_results = Test_dataset_causility(dataset, lags)

print(f"Average F-scores for {configs.dataset} is {grager_results}")

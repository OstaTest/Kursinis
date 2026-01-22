# This experiment compares the forecasting accuracy between the best average found in experiment1(individual and best overall) 
# and the best MA and period combo found that fits the out-of-sample data best

import utils
import mainFunctions
import pandas as pd

def evaluateInSampleMAPeriodCombos():
    all_results_2024_2025 = pd.read_excel("Experiment2_AllResults.xlsx")
    best_per_ticker_2014_2024 = pd.read_excel("Experiment1_BestPerTicker.xlsx")
    best_overall_2014_2024 = pd.read_excel("Experiment1_BestOverall.xlsx")
    best_per_ticker_2024_2025 = pd.read_excel("Experiment2_BestPerTicker.xlsx")

    best_per_ticker_2014_2024 = best_per_ticker_2014_2024[["Ticker", "Period", "Method"]]
    best_overall_2014_2024 = best_overall_2014_2024[["Period", "Method"]]

    filtered_results_for_best_overall = all_results_2024_2025[
        (all_results_2024_2025["Period"] == best_overall_2014_2024["Period"].iloc[0]) &
        (all_results_2024_2025["Method"] == best_overall_2014_2024["Method"].iloc[0])
    ]

    best_per_ticker_mae_list = []

    for _, row in best_per_ticker_2014_2024.iterrows():
        filtered = all_results_2024_2025[
            (all_results_2024_2025["Ticker"] == row["Ticker"]) &
            (all_results_2024_2025["Period"] == row["Period"]) &
            (all_results_2024_2025["Method"] == row["Method"])
        ]

        if not filtered.empty:
            mae_value = filtered["MAE"].values[0]
            best_per_ticker_mae_list.append((row["Ticker"], mae_value))

    final_table = pd.DataFrame()
    final_table["Ticker"] = best_per_ticker_2014_2024["Ticker"]
    final_table["Period"] = (best_per_ticker_2024_2025["Method"]+ "(" + best_per_ticker_2024_2025["Period"].astype(str) + ")")
    final_table["24-25 MAE"] = best_per_ticker_2024_2025["MAE"]
    final_table["14-24 individual MAE"] = final_table["Ticker"].map(dict(best_per_ticker_mae_list))
    final_table["14-24 all MAE"] = final_table["Ticker"].map(dict(zip(filtered_results_for_best_overall["Ticker"], filtered_results_for_best_overall["MAE"])))
    final_table["% Δ (24-25 vs 14-24 individual)"] = ((final_table["24-25 MAE"] - final_table["14-24 individual MAE"]) / final_table["14-24 individual MAE"]) * 100
    final_table["% Δ (24-25 vs 14-24 all)"] = ((final_table["24-25 MAE"] - final_table["14-24 all MAE"]) / final_table["14-24 all MAE"]) * 100

    utils.saveToExcel(final_table, "Experiment2_FinalTable")

mainFunctions.findBestMAAndPeriodCombo(utils.tickersArray, utils.movingAveragesArray, utils.OOSStartDate, utils.OOSEndDate)
evaluateInSampleMAPeriodCombos(utils.tickersArray, utils.OOSStartDate, utils.OOSEndDate)
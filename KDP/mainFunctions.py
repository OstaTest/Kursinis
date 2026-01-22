import pandas as pd
import utils


def findBestMAAndPeriodCombo(tickersArray, movingAveragesArray, startDate, endDate, minPeriod=2, maxPeriod=200):
    caller_name = utils.get_caller_filename()
    results = []
    periods = range(minPeriod, maxPeriod + 1)
    for ticker in tickersArray:
        print(f"Processing ticker: {ticker}")
        extendedStartDate = (pd.to_datetime(startDate) - pd.DateOffset(days=int(maxPeriod * 1.5)))
        data = utils.downloadData(ticker, extendedStartDate, endDate)
        if data.empty:
            print(f"No data for {ticker}, skipping.")
            continue
        data = utils.calculateVariance(data)
        data.dropna(inplace=True)

        realizedVar = data.loc[startDate:endDate, "Var"]

        for average in movingAveragesArray:
            for period in periods:
                forecastVar = utils.calculateMovingAverage(data, period, average).shift(1)
                mae = utils.calculate_forecast_mae(realizedVar, forecastVar)

                results.append({"Ticker": ticker, "Period": period, "Method": average, "MAE": mae})
    
    results_df = pd.DataFrame(results)

    best_per_ticker = (results_df.loc[results_df.groupby("Ticker")["MAE"].idxmin()].reset_index(drop=True))

    overall_scores = (results_df.groupby(["Period", "Method"])["MAE"].mean().reset_index())

    best_overall = overall_scores.loc[overall_scores["MAE"].idxmin()]

    utils.saveToExcel(results_df, f"{caller_name}_AllResults")
    utils.saveToExcel(best_per_ticker, f"{caller_name}_BestPerTicker")
    utils.saveToExcel(pd.DataFrame([best_overall]), f"{caller_name}_BestOverall")

    return best_per_ticker, best_overall


def computeDifferenceBetweenIndividualAndBestOverallMA():
    caller_name = utils.get_caller_filename()
    all_results = pd.read_excel(f"{caller_name}_AllResults.xlsx")
    best_per_ticker = pd.read_excel(f"{caller_name}_BestPerTicker.xlsx")
    best_overall = pd.read_excel(f"{caller_name}_BestOverall.xlsx")

    global_period = best_overall.loc[0, "Period"]
    global_method = best_overall.loc[0, "Method"]

    comparison_rows = []

    for _, row in best_per_ticker.iterrows():
        ticker = row["Ticker"]

        individual_mae = row["MAE"]

        global_mae = all_results.loc[(all_results["Ticker"] == ticker) & (all_results["Period"] == global_period) & (all_results["Method"] == global_method), "MAE"].values[0]

        comparison_rows.append({
            "Ticker": ticker,
            "Individual_Method": row["Method"],
            "Individual_Period": row["Period"],
            "Individual_MAE": individual_mae,
            "Global_Method": global_method,
            "Global_Period": global_period,
            "Global_MAE": global_mae,
            "MAE_Difference": global_mae - individual_mae,
            "MAE_Pct_Increase": (
                (global_mae - individual_mae) / individual_mae * 100
            )
        })

    comparison_df = pd.DataFrame(comparison_rows)

    utils.saveToExcel(
        comparison_df,
        f"{caller_name}_Individual_vs_Global_Comparison"
    )

    return comparison_df




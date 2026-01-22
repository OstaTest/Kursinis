import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import utils
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os


def experiment3(tickers, startDate, endDate, trainWindow=700):
    results = []

    for ticker in tickers:
        print(f"Running experiment for {ticker}")

        data = yf.download(ticker,
            start=pd.to_datetime(startDate) - pd.DateOffset(days=trainWindow * 1.5), # *1.5 because there are 252 trading days a year instead of 365
            end=pd.to_datetime(endDate) + pd.DateOffset(days=20),
            progress=False,
            auto_adjust=True
        )

        if data.empty:
            print(f"Download failed for {ticker}, skipping.")
            continue

        data = utils.calculateVariance(data)
        data.dropna(inplace=True)

        predicted_vars = []
        realized_vars = []
        
        prev_params = None

        for t in data.loc[startDate:endDate].index:

            train_returns = data["r"].loc[:t].iloc[-trainWindow:]

            if len(train_returns) < trainWindow:
                print("Data problem")
                continue

            train_returns = train_returns.dropna()

            model = arch_model(train_returns * 100, mean="Zero", vol="Garch", p=1, q=1, rescale=False)
            fit = model.fit(disp="off", options={"maxiter": 10000})
            if fit.convergence_flag != 0:
                print(f"Warning: convergence issue at {t}, reusing previous days parameters")
                if prev_params is None:
                    print("Crashed at the start, need to restart")
                garch_var = model.forecast(params=prev_params, horizon=1, reindex=False).variance.iloc[-1, 0] / 10000
            else:
                garch_var = fit.forecast(horizon=1).variance.iloc[-1, 0] / 10000
                prev_params = fit.params
                    
            

            next_day = t + pd.DateOffset(days=1)
            if next_day not in data.index:
                continue

            realized_var = data.loc[next_day, "r"] ** 2  # daily variance

            predicted_vars.append(garch_var)
            realized_vars.append(realized_var)

        if not predicted_vars:
            print(f"No valid forecasts for {ticker}, skipping.")
            continue

        mae = mean_absolute_error(realized_vars, predicted_vars)

        results.append({
            "Ticker": ticker,
            "GARCH_MAE": mae,
        })

    results_df = pd.DataFrame(results)
    results_df.sort_values("Ticker", inplace=True)

    utils.saveToExcel(results_df, f"Experiment3_{trainWindow}")

def load_best_individual_combos():
    df = pd.read_excel("Experiment1_BestPerTicker.xlsx")
    return df.set_index("Ticker")[["Method", "Period"]].to_dict("index")


def plot_daily_volatility_comparison(
    tickersArray,
    startDate,
    endDate,
    trainWindow=700,
):
    best_individual_combos = load_best_individual_combos()

    for ticker in tickersArray:
        print(f"Analysing ticker: {ticker}")

        best_combo = best_individual_combos[ticker]

        data = yf.download(
            ticker,
            start=pd.to_datetime(startDate) - pd.DateOffset(days=trainWindow * 3),
            end=pd.to_datetime(endDate) + pd.DateOffset(days=5),
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            raise ValueError(f"No data downloaded for {ticker}")

        data = utils.calculateVariance(data)
        data.dropna(inplace=True)

        data["common_var"] = (data["realized_var"].ewm(span=22, adjust=False).mean().shift(1))

        data["best_individual_var"] = (utils.calculateMovingAverage(pd.DataFrame({"Var": data["realized_var"]}), best_combo["Period"], best_combo["Method"]).shift(1))

        forecast_rows = []
        prev_params = None

        for t in data.loc[startDate:endDate].index:

            train_returns = data["r"].loc[:t].iloc[-trainWindow:]

            if len(train_returns) < trainWindow:
                continue

            model = arch_model(
                train_returns * 100,
                mean="Zero",
                vol="Garch",
                p=1,
                q=1,
                rescale=False
            )

            fit = model.fit(disp="off", options={"maxiter": 10000})

            if fit.convergence_flag != 0:
                if prev_params is None:
                    continue
                garch_var = (
                    model.forecast(
                        params=prev_params,
                        horizon=1,
                        reindex=False
                    ).variance.iloc[-1, 0] / 10000
                )
            else:
                garch_var = fit.forecast(horizon=1).variance.iloc[-1, 0] / 10000
                prev_params = fit.params

            next_day = t + pd.DateOffset(days=1)
            if next_day not in data.index:
                continue

            forecast_rows.append({
                "Date": next_day,
                "Stock": ticker,
                "Realized": float(data.loc[next_day, "realized_var"].iloc[0]),
                "GARCH": float(garch_var),
                "EMA": float(data.loc[next_day, "common_var"].iloc[0]),
                "BestIndividual": float(data.loc[next_day, "best_individual_var"].iloc[0]),
            })

        df_ticker = pd.DataFrame(forecast_rows)

        if not df_ticker.empty:
            plt.figure(figsize=(10, 5))

            plt.plot(df_ticker["Date"], df_ticker["Realized"], label="Tikrasis kintamumas", linewidth=2, alpha = 0.5)
            plt.plot(df_ticker["Date"], df_ticker["GARCH"], label="GARCH(1,1) prognozės", linestyle="--", linewidth=2)
            plt.plot(df_ticker["Date"], df_ticker["EMA"], label="EMA(22) prognozės", linestyle=":", linewidth=2)
            plt.plot(df_ticker["Date"], df_ticker["BestIndividual"], label=f'{best_combo["Method"]}({best_combo["Period"]}) prognozės', linestyle="-.", linewidth=2)

            plt.title(f"{ticker} - Tikrasis ir prognozuoti kintamumai")
            plt.ylabel("Kintamumas")
            plt.xlabel("Data")
            plt.legend()
            plt.tight_layout()

            individual_path = f"plots/{ticker}_volatility.png"
            plt.savefig(individual_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Saved individual plot: {individual_path}")



def compareResults():
    exp2_table = pd.read_excel("Experiment2_FinalTable.xlsx")
    garch_results = pd.read_excel("Experiment3_600.xlsx")

    df = pd.DataFrame()
    df["Ticker"] = garch_results["Ticker"]
    df["Garch MAE"] = garch_results["GARCH_MAE"]
    df["Individual MAE"] = exp2_table["14-24 individual MAE"]
    df["All MAE"] = exp2_table["14-24 all MAE"]
    df["Garch vs individual %"] = ((df["Individual MAE"] - df["Garch MAE"]) / df["Individual MAE"]) * 100
    df["Garch vs All %"] = ((df["All MAE"] - df["Garch MAE"]) / df["All MAE"]) * 100

    utils.saveToExcel(df, "Experiment3_FinalTable")

#experiment3(utils.tickersArray, utils.OOSStartDate, utils.OOSEndDate, trainWindow=600)
#compareResults()
plot_daily_volatility_comparison(["NVDA"], utils.OOSStartDate, utils.OOSEndDate, trainWindow=600)
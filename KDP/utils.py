import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import inspect
import os

tickersArray = ['AAPL', 'CVX', 'F', 'JNJ', 'JPM', 'LLY',  'MS', 'NVDA',  'TSLA', 'XOM']
movingAveragesArray = ['SMA', 'EMA', 'WMA', 'TMA', 'HMA']
startDates = ['2022-01-01', '2023-01-01', '2023-07-01']
OOSStartDate = '2024-01-01'
OOSEndDate = '2025-01-01'
endDate = '2024-01-01'
longer_start_date = "2014-01-01"
periods = [5, 10, 20, 50, 100, 200]


def get_caller_filename():
    stack = inspect.stack()
    # stack[0] = _get_caller_filename
    # stack[1] = findBestMAAndPeriodCombo
    # stack[2] = caller
    caller_frame = stack[2]
    module = inspect.getmodule(caller_frame.frame)

    if module and hasattr(module, "__file__"):
        return os.path.splitext(os.path.basename(module.__file__))[0]
    return "unknown_caller"


def calculate_forecast_mae(realized, forecast):
    aligned = pd.concat([realized, forecast], axis=1).dropna()
    aligned.columns = ["Actual", "Predicted"]
    
    if aligned.empty:
        return None
    
    return mean_absolute_error(aligned["Actual"], aligned["Predicted"])


def calculateVariance(dataframe):
    dataframe["r"] = np.log(dataframe["Close"] / dataframe["Close"].shift(1))
    dataframe["Var"] = dataframe["r"] ** 2
    return dataframe


def saveToExcel(experimentResults, excelName, saveAsTable=False):
    df = pd.DataFrame(experimentResults)
    df.to_excel(f"{excelName}.xlsx", index = False, engine = 'openpyxl')

    if saveAsTable:
        df['StartDate'] = pd.to_datetime(df['StartDate'])
        df.sort_values(['StartDate', 'Ticker', 'Period'], inplace=True)
        pivot_table = df.pivot_table(
            index='Ticker',
            columns=['StartDate', 'Period'],
            values='BestMethod',
            aggfunc='first'
        )

        pivot_table.columns = [
            f"{pd.to_datetime(start).strftime('%Y-%m-%d')}_{period}" 
            for start, period in pivot_table.columns
        ]

        pivot_table.reset_index(inplace=True)

        pivot_table.to_excel(f"{excelName}Table.xlsx", index=False, engine='openpyxl')


def downloadData(ticker, startDate, endDate):
    data = yf.download(ticker, start=startDate, end=endDate, progress=False, auto_adjust=True)
    return data


def debugPlot(movingAverageData, dataframe, period, averageType, ticker):
    plt.figure(figsize=(10,7))
    plt.plot(dataframe['Close'], label=f'{ticker} Stock Price', color='blue')
    plt.plot(movingAverageData, label=f'{averageType} (n={period})', color='green', linestyle='--')
    plt.title(f'{ticker} Stock Price with {averageType} (n={period})')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def WMA(dataframe, period):
    weights = np.arange(1, period + 1)
    wma = dataframe.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return wma


def calculateMovingAverage(dataframe, period, name):
    global printing
    match name:
        case 'SMA':
            movingAverageValues =  dataframe['Var'].rolling(window=period).mean()
        case 'EMA':
            #Calling exponential weight calculations and then calculating the average of it
            movingAverageValues =  dataframe['Var'].ewm(span=period, adjust=False).mean()
        case 'WMA':
            #Create weights from 1 to period + 1 values: p = 5, [1, 2, 3, 4, 5]
            weights = np.arange(1, period + 1)
            #Calculating dot product between the weights and the period amount of last values; then dividing it by all the summed weights as in the formula
            movingAverageValues = dataframe['Var'].rolling(window=period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        case 'TMA':
            halfN = period // 2
            halfSma = dataframe['Var'].rolling(window=period).mean()
            movingAverageValues = halfSma.rolling(window = halfN).mean()
        case 'HMA':
            wmaHalfPeriod = WMA(dataframe['Var'], period // 2)
            wmaFullPeriod = WMA(dataframe['Var'], period)
            movingAverageValues = WMA(2 * wmaHalfPeriod - wmaFullPeriod, int(np.sqrt(period)))
        case _:
            print("Unknown moving average type")

    return movingAverageValues
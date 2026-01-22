# This experiment finds best moving averages for multiple start dates with set periods; 
# Secondly it finds best moving average and period combo for each ticker individually and a MA and period combo that is best over all of the tickers over a period of 10 years

import utils
import mainFunctions
import pandas as pd

def findBestMAWithSetPeriodsOverMultipleStartDates(startDates, tickersArray, periods, movingAveragesArray, endDate):
    resultsList = []

    for ticker in tickersArray:
        print(f"Processing ticker: {ticker}")

        extendedStartDate = pd.to_datetime(min(startDates)) - pd.DateOffset(days=int(max(periods) * 1.5))
        extendedEndDate = pd.to_datetime(endDate) + pd.DateOffset(days=1)

        data = utils.downloadData(ticker, extendedStartDate, extendedEndDate)
        if data.empty:
            print(f"No data for {ticker}, skipping.")
            continue
        
        data = utils.calculateVariance(data)
        data.dropna(inplace=True)

        realizedVarianceDictionary = {
            startDate: data.loc[pd.to_datetime(startDate):pd.to_datetime(endDate), "Var"]
            for startDate in startDates
        }

        for startDate in startDates:
            realizedVar = realizedVarianceDictionary[startDate]

            for period in periods:
                bestMae = float("inf")
                bestMse = None
                bestAverage = None

                for average in movingAveragesArray:
                    forecastVar = utils.calculateMovingAverage(data, period, average).shift(1)

                    mae = utils.calculate_forecast_mae(realizedVar, forecastVar)

                    if mae < bestMae:
                        bestMae = mae
                        bestAverage = average

                resultsList.append({"StartDate": startDate, "Ticker": ticker, "Period": period, "BestMethod": bestAverage, "MAE": bestMae, "MSE": bestMse})

    experiment1Results = pd.DataFrame(resultsList)
    utils.saveToExcel(experiment1Results, "Experiment1", saveAsTable=True)

findBestMAWithSetPeriodsOverMultipleStartDates(utils.startDates, utils.tickersArray, utils.periods, utils.movingAveragesArray, utils.endDate)
mainFunctions.findBestMAAndPeriodCombo(utils.tickersArray, utils.movingAveragesArray, utils.longer_start_date, utils.endDate)
mainFunctions.computeDifferenceBetweenIndividualAndBestOverallMA()
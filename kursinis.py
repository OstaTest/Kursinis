import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotExp2(bestMovingAverageForPeriod):
    #Collect all periods their error values and the best moving average for each of them in 3 separate lists
    periods = list(bestMovingAverageForPeriod.keys())
    errors = [data['error'] for data in bestMovingAverageForPeriod.values()]
    averages = [data['average'] for data in bestMovingAverageForPeriod.values()]

    #Make a colour map, to assign each colour to a unique moving average, the colors list will have each colour assigned to each period based on its moving average
    uniqueAverages = sorted(set(averages))
    colorMap = plt.get_cmap('viridis', len(uniqueAverages))
    colors = [uniqueAverages.index(avg) for avg in averages]


    plt.figure(figsize=(12, 6))
    plt.title("Geriausiai prognozuojantis slenkantis vidurkis su kiekvienu periodu")
    plt.xlabel("Periodai")
    plt.ylabel("Prognozavimo atsilikimo vertė nuo faktinių duomenų")

    #Plot periods on x axis and errors on y axis, and provide colours
    plt.scatter(periods, errors, c=colors, cmap=colorMap, s=50, edgecolor='k')

    #Puts a text box around the best moving average + period pair; xy provides the place where to put the text box(next to the point); xytext puts the location of the text; bbox adds the box around the text
    minIndex = errors.index(min(errors))
    plt.annotate(f'Periodas: {periods[minIndex]}\nAtsilikimas: {errors[minIndex]:.2f}\nVidurkis: {averages[minIndex]}',
                 xy=(periods[minIndex], errors[minIndex]),
                 xytext=(periods[minIndex] + 10, errors[minIndex]),
                 fontsize=8, bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))


    #Plot nothing then assign colour to the "invisible point"; Assign a label to the point(to the colour) so it would appear on the legend.
    for i, avg in enumerate(uniqueAverages):
        plt.scatter([], [], c=colorMap(i), label=f'{avg}', edgecolor='k')

    plt.legend(title="Slenkantis Vidurkis", loc='upper left')

    plt.show()



def experiment2(startDates, tickersArray, periods, movingAveragesArray, maxPeriod=200):
    bestMovingAverageForPeriod = {}
    #From 2 to 200; for 10 stocks; for 3 start dates; from 2022-01-01 is 730 days; from 2023-01-01 is 365 days; from 2023-07-01 is 184 days
    #In total there are 1279 days for 10 stocks there are 12790 days.
    #for period in range(2, maxPeriod + 1):
    for period in periods:
        print(period)
        minError = float('inf')
        bestMovingAverage = None

        for average in movingAveragesArray:
            totalError = 0

            for ticker in tickersArray:
                for startDate in startDates:
                    extendedStartDate = pd.to_datetime(startDate) - pd.DateOffset(days = max(periods) * 2)
                    extendedEndDate = pd.to_datetime(endDate) + pd.DateOffset(days = max(periods))
                    data = downloadData(ticker, extendedStartDate, extendedEndDate)
                    data['TR'] = calculateTR(data)
                    movingAverageValues = calculateMovingAverage(data, period, average)
                    pastMovingAverageValues = movingAverageValues.shift(-1)
                    tempData = data[(data.index >= startDate) & (data.index <= endDate)]
                    tempPastMovingAverageValues = pastMovingAverageValues[(pastMovingAverageValues.index >= startDate) & (pastMovingAverageValues.index <= endDate)]
                    differences = tempData['TR'].dropna() - tempPastMovingAverageValues.dropna()
                    sumOfDifferences = differences.abs().sum()
                    totalError += sumOfDifferences
            if totalError < minError:
                minError = totalError
                bestMovingAverage = average
        
        bestMovingAverageForPeriod[period] = {'average': bestMovingAverage, 'error': minError}

    plotExp2(bestMovingAverageForPeriod)
    
def experiment1(startDates, tickersArray, periods, movingAveragesArray):
    performanceResults = {}
    for startDate in startDates:

        performanceResults[startDate] = {}

        for ticker in tickersArray:
            #Making the dates longer so that shift would not give NaN for first value and so the first period values would not be NaN
            extendedStartDate = pd.to_datetime(startDate) - pd.DateOffset(days = max(periods) * 2)
            extendedEndDate = pd.to_datetime(endDate) + pd.DateOffset(days = max(periods))
            data = downloadData(ticker, extendedStartDate, extendedEndDate)
            data['TR'] = calculateTR(data)
        
            resultsForTicker = {}

            #Iterating through all the different periods
            for period in periods:

                minSum = float('inf')
                bestAverage = None

                #Iterating through all the different averages
                for average in movingAveragesArray:
                    movingAverageValues = calculateMovingAverage(data, period, average)
                    #Shifting moving average values by 1 into the past so we could compare the todays true range value and yesterdays moving average value
                    pastMovingAverageValues = movingAverageValues.shift(-1)
                    #Cutting off the additional lines we downloaded for making sure that no NaN values end up in the calculation
                    tempData = data[(data.index >= startDate) & (data.index <= endDate)]
                    tempPastMovingAverageValues = pastMovingAverageValues[(pastMovingAverageValues.index >= startDate) & (pastMovingAverageValues.index <= endDate)]
                    #Finding the difference between todays TR values and yesterdays moving average values so we could see how accuratelly does each moving average forecast the true range value
                    differences = tempData['TR'].dropna() - tempPastMovingAverageValues.dropna()
                    #Summing up all the values to see which moving average predicted the TR value the best throughout the whole date range
                    sumOfDifferences = differences.abs().sum()

                    if sumOfDifferences < minSum:
                        minSum = sumOfDifferences
                        bestAverage = average

                #Making a dictionary to store best moving average name and its sum for each period
                resultsForTicker[period] = {'bestMovingAverage': bestAverage, 'minSum': minSum}

            #Putting values inside another dictionary which will hold the results for ticker values for each ticker 
            performanceResults[startDate][ticker] = resultsForTicker

    resultsList = []

    for startDate, results in performanceResults.items():
        for ticker, periodsResult in results.items():
            for period, result in periodsResult.items():
                resultsList.append({'StartDate': startDate,
                                    'Ticker': ticker,
                                    'Period': period,
                                    'BestMethod': result['bestMovingAverage'],
                                    'MinSum': result['minSum']})

    experiment1Results = pd.DataFrame(resultsList)
    saveToExcel(experiment1Results)


def saveToExcel(experimentResults):
    df = pd.DataFrame(experimentResults)
    df.to_excel('results.xlsx', index = False, engine = 'openpyxl')

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

def downloadData(ticker, startDate, endDate):
    data = yf.download(ticker, start = startDate, end = endDate, progress = False)
    return data

def calculateTR(dataframe):
    dataframe['Previous Close'] = dataframe['Close'].shift(1)
    dataframe['High Low'] = dataframe['High'] - dataframe['Low']
    dataframe['High Previous Close'] = abs(dataframe['High'] - dataframe['Previous Close'])
    dataframe['Low Previous Close'] = abs(dataframe['Low'] - dataframe['Previous Close'])
    dataframe['TR'] = dataframe[['High Low', 'High Previous Close', 'Low Previous Close']].max(axis=1)
    dataframe.drop(columns=['High Low', 'High Previous Close', 'Low Previous Close', 'Previous Close'], inplace=True)  # Drop additional columns
    return dataframe['TR']

def WMA(dataframe, period):
    weights = np.arange(1, period + 1)
    wma = dataframe.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return wma

def calculateMovingAverage(dataframe, period, name):
    match name:
        case 'SMA':
            movingAverageValues =  dataframe['TR'].rolling(window=period).mean()
        case 'EMA':
            movingAverageValues =  dataframe['TR'].ewm(span=period, adjust=False).mean()
        case 'WMA':
            weights = np.arange(1, period + 1)
            movingAverageValues = dataframe['TR'].rolling(window=period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        case 'TMA':
            halfN = period // 2
            dataframe['halfSma'] = dataframe['TR'].rolling(window = period).mean()
            movingAverageValues = dataframe['halfSma'].rolling(window = halfN).mean()
        case 'HMA':
            wmaHalfPeriod = WMA(dataframe['TR'], period // 2)
            wmaFullPeriod = WMA(dataframe['TR'], period)
            movingAverageValues = WMA(2 * wmaHalfPeriod - wmaFullPeriod, int(np.sqrt(period)))

    return movingAverageValues


#AAPL - apple(technology sector); NVDA - nvidia(technology sector); F - ford motor company(automotive sector); TSLA - tesla(automotive sector)
#XOM - ExxonMobil(energy sector); CVX - chevron corporation(energy sector); JNJ - Johnson & Johnson(healthcare sector); LLY - Eli Lily and Company(healthcare sector);
#JPM - JPMorgan chase % co. (finance/banking sector); MS - Morgan stanley(finance/banking sector); 
tickersArray = ['AAPL', 'NVDA', 'F', 'TSLA', 'XOM', 'CVX', 'JNJ', 'LLY', 'JPM', 'MS']
#SMA - simple moving average, EMA - exponential moving average, WMA - weighted moving average, TMA - triangular moving average, HMA - Hull moving average
movingAveragesArray = ['SMA', 'EMA', 'WMA', 'TMA', 'HMA']
#startDate = '2022-01-01'
startDates = ['2022-01-01', '2023-01-01', '2023-07-01']
endDate = '2024-01-01'
periods = [5, 10, 20, 50, 100, 200]

#experiment1(startDates, tickersArray, periods, movingAveragesArray)
experiment2(startDates, tickersArray, periods, movingAveragesArray)





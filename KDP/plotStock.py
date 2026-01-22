import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import utils

def plot_candlestick_graph():
    data = utils.downloadData('NVDA', '2023-06-01', '2024-01-01')
    data = utils.calculateVariance(data)

    data.columns = data.columns.droplevel(1)
    date_format = mdates.DateFormatter('%m-%d')
    fig, ax = mpf.plot(data, type='candle', style='charles', title= 'Nvidia laiko eilutė nuo 2023-06-01 iki 2024-01-01', figratio=(12,5), figscale=1, xlabel = 'Data', ylabel = 'Kaina', returnfig=True)
    ax[0].xaxis.set_major_formatter(date_format)
    for label in ax[0].get_xticklabels():
        label.set_rotation(0)
    mpf.show()

def plot_price_and_variance():
    data = utils.downloadData('NVDA', '2023-06-01', '2024-01-01')
    data = utils.calculateVariance(data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True, sharex=True)

    # Price
    ax1.plot(data.index, data['Close'])
    ax1.set_ylabel('Kaina')
    ax1.grid(True)

    # Day-to-day variance
    ax2.plot(data.index, data['daily_variance'])
    ax2.set_ylabel('Dienos kintamumo vertės')
    ax2.set_xlabel('Data')
    ax2.grid(True)

    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.show()

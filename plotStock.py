import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

data = yf.download('NVDA', '2023-06-01', '2024-01-01')
# Plot candlestick
mpf.plot(data, 
         type='candle', 
         style='charles', 
         #title= 'Nvidia laiko eilutė nuo 2023-06-01 iki 2024-01-01', 
         figratio=(15,4),
         figscale=1,
         xlabel = 'Data',
         ylabel = 'Kaina'
         )

# Plot linear price graph
plt.figure(figsize=(15, 4))
plt.plot(data.index, data['Close'], label='NVDA Closing Price', color='blue')

#plt.title('NVDA Akcijos kaina (Birželis 2023 - Sausis 2024)')
plt.xlabel('Data')
plt.ylabel('Kaina')
#plt.legend()
plt.grid(True)
plt.show()

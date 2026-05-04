
# TIME SERIES ANALYSIS

#Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error

#Load Time Series Data

df = pd.read_csv("./Datasets/forecast.csv")

#Clean column names
df.columns = df.columns.str.strip()


#Convert to datetime

df['Time Date'] = pd.to_datetime(df['Time Date'])

#Rename Value Column with Sale and Time Date column with Date
df.rename(columns={'Value': 'Sales'}, inplace=True)




#Set index
df.set_index('Time Date', inplace=True)

#Sort index
df = df.sort_index()
print(df.shape)



#Handle misssing values
df['Sales'] = df['Sales'].bfill()

print(df.head())


#Plot Time Series

plt.figure(figsize=(10,5))
plt.plot(df['Sales'])
plt.title("Time Series Data")
plt.xlabel("Time Date ")
plt.ylabel("Sales")
plt.show()

#Decompose Time Series

decomposition = seasonal_decompose(df['Sales'], model='additive', period=4)

decomposition.plot()
plt.show()

#Moving Average

df['MA_3'] = df['Sales'].rolling(window=3).mean()

plt.plot(df['Sales'], label='original')
plt.plot(df['MA_3'], label='Moving Average')
plt.legend()
plt.show()

#Exponential Smoothing

model = ExponentialSmoothing(df['Sales'], trend='add', seasonal=None)
fit = model.fit()

df['ES'] = fit.fittedvalues

plt.plot(df['Sales'], label= 'Original')
plt.plot(df['ES'], label= 'Smoothed')
plt.legend()
plt.show()


#Train-Test Split

train = df.iloc[:-12]
test = df.iloc[-12:]

#ARIMA Model

model = ARIMA(train['Sales'], order=(1,1,1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))


#Evaluate Model(RMSE)

rmse = np.sqrt(mean_squared_error(test['Sales'], forecast))
print("RMSE:", rmse)


#Plot Forecast

plt.plot(train['Sales'], label='Train')
plt.plot(test['Sales'], label='Test')
plt.plot(forecast, label='Forecast')

plt.legend()
plt.show()









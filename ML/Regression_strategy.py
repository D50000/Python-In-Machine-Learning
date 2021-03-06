# ============================================================
# node-binance-api
# https://binance-docs.github.io/apidocs/spot/en/#market-data-endpoints
# ============================================================
# Copyright 2020, Eddie Hsu (D50000)
# Released MySelf
# ============================================================

import requests
import datetime

import pandas as pd
print('pandas: ' + pd.__version__)
import numpy as np
print ('numpy: ' + np.version.version)
import matplotlib.pyplot as plt
import talib

################### data preprocessing
r1 = requests.get('http://api.binance.com/api/v3/time').json()
period = 86400 * 360 * 1000  # ms
startTime = r1['serverTime'] - period
interval = '6h'
raw_Data = requests.get('https://api.binance.com/api/v1/klines?symbol=ETHUSDT&interval=' + interval).json()
# note: 6h and 1d is the best predict rate.
# print(r2)

#   [
#     1499040000000,      // #0 Open time
#     "0.01634790",       // #1 Open
#     "0.80000000",       // #2 High
#     "0.01575800",       // #3 Low
#     "0.01577100",       // #4 Close
#     "148976.11427815",  // #5 Volume
#     1499644799999,      // #6 Close time
#     "2434.19055334",    // #7 Quote asset volume
#     308,                // #8 Number of trades
#     "1756.87402397",    // #9 Taker buy base asset volume
#     "28.46694368",      // #10 Taker buy quote asset volume
#     "17928899.62484339" // #11 Ignore.
#   ]


date_Array = []
Open_Array = []
High_Array = []
Low_Array = []
Close_Array = []
Volume_Array = []
latest_data = None
for i, c in enumerate(raw_Data):
    start = str(c[0])
    newTime = int(start[0:10])
    date = datetime.datetime.fromtimestamp(newTime).isoformat()
    if i == len(raw_Data)-1:
        print(date)
        print('Last data: ' + str(c))
        latest_data = date
    date_Array.append(date)
    Open_Array.append(float(c[1]))
    High_Array.append(float(c[2]))
    Low_Array.append(float(c[3]))
    Close_Array.append(float(c[4]))
    Volume_Array.append(float(c[5]))
# print(Volume_Array)

df_numpy = {
    'date': date_Array,
    'Open': Open_Array,
    'High': High_Array,
    'Low': Low_Array,
    'Close': Close_Array,
    'Volume': Volume_Array
    }
df = pd.DataFrame(data=df_numpy)
df.index = pd.to_datetime(df.date.astype(np.str))

price = df['Close']
df['sma'] = talib.SMA(price.values, 7)
df['ema'] = talib.EMA(price.values, 7)
df['wma'] = talib.WMA(price.values, 7)
df['ADXR'] = talib.ADXR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=3)
df['mom'] = talib.MOM(price.values, 10)
df['K_percent'], df['D_percent'] = talib.STOCHF(df['High'].values, df['Low'].values, df['Close'].values)
df['rsi'] = talib.RSI(price.values, 10)
df['macd'], dif, dem = talib.MACD(price.values)
df['W_percent'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values)
df['cci'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values)
df['ADO'] = talib.ADOSC(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values)
# print(ADO)

# price.pct_change().head()
df['price_mov'] = price.pct_change().shift(-1)
df=df.dropna()
print(df)


################### split data to train and test group
n_sample = df.shape[0]
n_train = np.int(n_sample*0.5)
train = df.iloc[:n_train,:]
test = df.iloc[n_train:,:]
# print(train)

# print(df.columns)
train_X = train[['sma', 'ema', 'wma','ADXR', 'mom', 'K_percent', 'D_percent', 'rsi', 'macd', 'W_percent', 'cci', 'ADO']]
train_Y = np.array(train[['price_mov']])

test_X = test[['sma', 'ema', 'wma','ADXR', 'mom', 'K_percent', 'D_percent', 'rsi', 'macd', 'W_percent', 'cci', 'ADO']]
test_Y = np.array(test[['price_mov']])
print(train_X)

################### build the LinearRegression model 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
clf = LinearRegression()
clf.fit(train_X, train_Y.ravel())
print('intercept:',clf.intercept_)
print('coefficient:',clf.coef_)
print(clf.predict(test_X))
# Evaluate the model
mse = np.mean((clf.predict(test_X) - train_Y) ** 2)
r_squared = clf.score(train_X, train_Y)
adj_r_squared = 1 - (1 - r_squared) * (train_X.shape[0] - 1) / (train_X.shape[0] - train_X.shape[1] - 1)


print('Mean squared error: ' + str(mse))
print('R-squared: ' + str(r_squared))
print('Adjusted R-squared: ' + str(adj_r_squared))
print('p-value: '+ str(f_regression(train_X, train_Y.ravel())[1]))


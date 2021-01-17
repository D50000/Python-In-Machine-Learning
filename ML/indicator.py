# ============================================================
# node-binance-api
# https://binance-docs.github.io/apidocs/futures/en/#market-data-endpoints
# https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=1h&limit=1500
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

################### data preprocessing configuration
r1 = requests.get('https://fapi.binance.com/fapi/v1/time').json()
period = 86400 * 360 * 1000  # ms
startTime = r1['serverTime'] - period
interval = '5m'
raw_Data = requests.get('https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&limit=1500&interval=' + interval).json()


# [
#   [
#     1499040000000,      // 開盤時間
#     "0.01634790",       // 開盤價格
#     "0.80000000",       // 最高價格
#     "0.01575800",       // 最低價格
#     "0.01577100",       // 收盤價格
#     "148976.11427815",  // 成交數量
#     1499644799999,      // 收盤時間
#     "2434.19055334",    // 成交金额
#     308,                // 成交筆數
#     "1756.87402397",    // taker成交數量
#     "28.46694368",      // taker成交金额
#     "17928899.62484339" // 忽略
#   ]
# ]


date_Array = []
Open_Array = []
High_Array = []
Low_Array = []
Close_Array = []
Volume_Array = []
Total_Array = []
Order_Array = []
Taker_Volume_Array = []
Taker_Total_Array = []
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
    Total_Array.append(float(c[6]))
    Order_Array.append(float(c[7]))
    Taker_Volume_Array.append(float(c[8]))
    Taker_Total_Array.append(float(c[9]))

df_numpy = {
    'date': date_Array,
    'Open': Open_Array,
    'High': High_Array,
    'Low': Low_Array,
    'Close': Close_Array,
    'Volume': Volume_Array,
    'Total': Total_Array,
    'Order': Order_Array,
    'Taker_Volume': Taker_Volume_Array,
    'Taker_Total': Taker_Total_Array
    }
df = pd.DataFrame(data=df_numpy)
df.index = pd.to_datetime(df.date.astype(np.str))
price = df['Close']


"""
################### SMA
df['sma3'] = talib.SMA(price.values, 3)
df['sma6'] = talib.SMA(price.values, 6)
df['sma12'] = talib.SMA(price.values, 12)
df['sma24'] = talib.SMA(price.values, 24)
df['sma48'] = talib.SMA(price.values, 48)
df['sma96'] = talib.SMA(price.values, 96)
df['sma144'] = talib.SMA(price.values, 144)
df['sma288'] = talib.SMA(price.values, 288)

################### convert technical data to trend prediction signal data
df['sma3_sig'] = (df['Close']>=df['sma3']).astype(np.int).replace(0, -1)
df['sma6_sig'] = (df['Close']>=df['sma6']).astype(np.int).replace(0, -1)
df['sma12_sig'] = (df['Close']>=df['sma12']).astype(np.int).replace(0, -1)
df['sma24_sig'] = (df['Close']>=df['sma24']).astype(np.int).replace(0, -1)
df['sma48_sig'] = (df['Close']>=df['sma48']).astype(np.int).replace(0, -1)
df['sma96_sig'] = (df['Close']>=df['sma96']).astype(np.int).replace(0, -1)
df['sma144_sig'] = (df['Close']>=df['sma144']).astype(np.int).replace(0, -1)
df['sma288_sig'] = (df['Close']>=df['sma288']).astype(np.int).replace(0, -1)

################### EMA
df['ema3'] = talib.EMA(price.values, 3)
df['ema6'] = talib.EMA(price.values, 6)
df['ema12'] = talib.EMA(price.values, 12)
df['ema24'] = talib.EMA(price.values, 24)
df['ema48'] = talib.EMA(price.values, 48)
df['ema96'] = talib.EMA(price.values, 96)
df['ema144'] = talib.EMA(price.values, 144)
df['ema288'] = talib.EMA(price.values, 288)

################### convert technical data to trend prediction signal data
df['ema3_sig'] = (df['Close']>=df['ema3']).astype(np.int).replace(0, -1)
df['ema6_sig'] = (df['Close']>=df['ema6']).astype(np.int).replace(0, -1)
df['ema12_sig'] = (df['Close']>=df['ema12']).astype(np.int).replace(0, -1)
df['ema24_sig'] = (df['Close']>=df['ema24']).astype(np.int).replace(0, -1)
df['ema48_sig'] = (df['Close']>=df['ema48']).astype(np.int).replace(0, -1)
df['ema96_sig'] = (df['Close']>=df['ema96']).astype(np.int).replace(0, -1)
df['ema144_sig'] = (df['Close']>=df['ema144']).astype(np.int).replace(0, -1)
df['ema288_sig'] = (df['Close']>=df['ema288']).astype(np.int).replace(0, -1)
"""

################### WMA
df['wma3'] = talib.WMA(price.values, 3)
df['wma6'] = talib.WMA(price.values, 6)
df['wma12'] = talib.WMA(price.values, 12)
df['wma24'] = talib.WMA(price.values, 24)
df['wma48'] = talib.WMA(price.values, 48)
df['wma96'] = talib.WMA(price.values, 96)
df['wma144'] = talib.WMA(price.values, 144)
df['wma288'] = talib.WMA(price.values, 288)

################### convert technical data to trend prediction signal data
df['wma3_sig'] = (df['Close']>=df['wma3']).astype(np.int).replace(0, -1)
df['wma6_sig'] = (df['Close']>=df['wma6']).astype(np.int).replace(0, -1)
df['wma12_sig'] = (df['Close']>=df['wma12']).astype(np.int).replace(0, -1)
df['wma24_sig'] = (df['Close']>=df['wma24']).astype(np.int).replace(0, -1)
df['wma48_sig'] = (df['Close']>=df['wma48']).astype(np.int).replace(0, -1)
df['wma96_sig'] = (df['Close']>=df['wma96']).astype(np.int).replace(0, -1)
df['wma144_sig'] = (df['Close']>=df['wma144']).astype(np.int).replace(0, -1)
df['wma288_sig'] = (df['Close']>=df['wma288']).astype(np.int).replace(0, -1)

# price.pct_change().head()
df['price_mov'] = np.sign(price.pct_change().shift(-1))
df=df.dropna()
print(df)


################### split data to train and test group
n_sample = df.shape[0]
n_train = np.int(n_sample*0.7)
train = df.iloc[:n_train,:]
test = df.iloc[n_train:,:]
# print(train)

# print(df.columns)
train_X = train[['Open', 'High', 'Low', 'Close', 'Taker_Volume',
                 'wma3_sig', 'wma6_sig', 'wma12_sig', 'wma24_sig', 'wma48_sig', 'wma96_sig', 'wma144_sig', 'wma288_sig']]
train_Y = np.array(train[['price_mov']])

test_X = test[['Open', 'High', 'Low', 'Close', 'Taker_Volume',
                 'wma3_sig', 'wma6_sig', 'wma12_sig', 'wma24_sig', 'wma48_sig', 'wma96_sig', 'wma144_sig', 'wma288_sig']]
test_Y = np.array(test[['price_mov']])

# ################### build the LinearRegression model 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
model = LinearRegression()
model.fit(train_X, train_Y)
print('intercept:',model.intercept_)
print('coefficient:',model.coef_)
# print(model.predict(test_X))
min_max_scaler = MinMaxScaler((-1,1))
print(min_max_scaler.fit_transform(model.predict(test_X)))
# Evaluate the model
mse = np.mean((model.predict(test_X) - test_Y) ** 2)
r_squared = model.score(test_X, test_Y)
adj_r_squared = r_squared - (1 - r_squared) * (test_X.shape[1] / (test_X.shape[0] - test_X.shape[1] - 1))


print('Mean squared error: ' + str(mse))
print('R-squared: ' + str(r_squared))
print('Adjusted R-squared: ' + str(adj_r_squared))
print('p-value: '+ str(f_regression(test_X, test_Y.ravel())[1]))
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
interval = '1h'
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
df['sma5'] = talib.SMA(price.values, 5)
df['ema5'] = talib.EMA(price.values, 5)
df['sma10'] = talib.SMA(price.values, 10)
df['ema10'] = talib.EMA(price.values, 10)
df['sma20'] = talib.SMA(price.values, 20)
df['ema20'] = talib.EMA(price.values, 20)
df['sma30'] = talib.SMA(price.values, 30)
df['ema30'] = talib.EMA(price.values, 30)
df['sma50'] = talib.SMA(price.values, 50)
df['ema50'] = talib.EMA(price.values, 50)
df['sma100'] = talib.SMA(price.values, 100)
df['ema100'] = talib.EMA(price.values, 100)
df['sma200'] = talib.SMA(price.values, 200)
df['ema200'] = talib.EMA(price.values, 200)


################### convert technical data to trend prediction signal data
df['sma5_sig'] = (df['Close']>=df['sma5']).astype(np.int).replace(0, -1)
df['ema5_sig'] = (df['Close']>=df['ema5']).astype(np.int).replace(0, -1)
df['sma10_sig'] = (df['Close']>=df['sma10']).astype(np.int).replace(0, -1)
df['ema10_sig'] = (df['Close']>=df['ema10']).astype(np.int).replace(0, -1)
df['sma20_sig'] = (df['Close']>=df['sma20']).astype(np.int).replace(0, -1)
df['ema20_sig'] = (df['Close']>=df['ema20']).astype(np.int).replace(0, -1)
df['sma30_sig'] = (df['Close']>=df['sma30']).astype(np.int).replace(0, -1)
df['ema30_sig'] = (df['Close']>=df['ema30']).astype(np.int).replace(0, -1)
df['sma50_sig'] = (df['Close']>=df['sma50']).astype(np.int).replace(0, -1)
df['ema50_sig'] = (df['Close']>=df['ema50']).astype(np.int).replace(0, -1)
df['sma100_sig'] = (df['Close']>=df['sma100']).astype(np.int).replace(0, -1)
df['ema100_sig'] = (df['Close']>=df['ema100']).astype(np.int).replace(0, -1)
df['sma200_sig'] = (df['Close']>=df['sma200']).astype(np.int).replace(0, -1)
df['ema200_sig'] = (df['Close']>=df['ema200']).astype(np.int).replace(0, -1)

# price.pct_change().head()
# df['price_mov'] = np.sign(price.pct_change().shift(-1))  # predict Long/Short
df['price_mov'] = price.pct_change().shift(-1)         # predict change %
df=df.dropna()
print(df)


################### split data to train and test group
n_sample = df.shape[0]
n_train = np.int(n_sample*0.5)
train = df.iloc[:n_train,:]
test = df.iloc[n_train:,:]
# print(train)

# print(df.columns)
train_X = train[['Open', 'High', 'Low', 'Close', 'Volume', 'Total', 'Order', 'Taker_Volume', 'Taker_Total',
                 'sma5_sig', 'ema5_sig', 'sma10_sig', 'ema10_sig', 'sma20_sig', 'ema20_sig', 'sma30_sig', 'ema30_sig', 'sma50_sig', 'ema50_sig', 'sma100_sig', 'ema100_sig', 'sma200_sig', 'ema200_sig']]
train_Y = np.array(train[['price_mov']])

test_X = test[['Open', 'High', 'Low', 'Close', 'Volume', 'Total', 'Order', 'Taker_Volume', 'Taker_Total',
               'sma5_sig', 'ema5_sig', 'sma10_sig', 'ema10_sig', 'sma20_sig', 'ema20_sig', 'sma30_sig', 'ema30_sig', 'sma50_sig', 'ema50_sig', 'sma100_sig', 'ema100_sig', 'sma200_sig', 'ema200_sig']]
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

# To do:
# regression trainX: +/- > predict y > +/- , result do "normalization" then "binary"


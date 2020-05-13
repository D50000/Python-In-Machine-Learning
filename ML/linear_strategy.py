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
        print(c)
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
df['price_mov'] = np.sign(price.pct_change().shift(-1))
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

# ################### normalization the indicators
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(train_X)
train_X_scale = scaler.transform(train_X)
test_X_scale = scaler.transform(test_X)
# # print(test_X_scale[:3])

"""
################### build the LogisticRegression model 
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_X_scale, train_Y.ravel())
train_Y_predict = clf.predict(train_X_scale)
test_Y_predict = clf.predict(test_X_scale)
# evaluate the model
# T/F accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix(train_Y, train_Y_predict)
# print(confusion_matrix(train_Y, train_Y_predict))
print('-------------------- LogisticRegression model --------------------')
p_true = pd.DataFrame(confusion_matrix(train_Y, train_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
print()
p_true = pd.DataFrame(confusion_matrix(test_Y, test_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
print()
# Precision Recall
from sklearn.metrics import classification_report
report = classification_report(np.array(train_Y), list(train_Y_predict), output_dict=True)
print(pd.DataFrame(report))
print()
test_report = classification_report(np.array(test_Y), list(test_Y_predict), output_dict=True)
print(pd.DataFrame(test_report))
"""

# ################### build the LinearRegression model 
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(train_X_scale, train_Y.ravel())
print('intercept:',clf.intercept_)
print('coefficient:',clf.coef_)



"""
# ################### convert to trend prediction data
# df['sma34_sig'] = (df['Close']>=df['sma34']).astype(np.int)
# df['sma34_sig'].replace(0, -1, inplace=True)

# # df['sma200_sig'] = (df['Close']>=df['sma200']).astype(np.int)
# # df['sma200_sig'].replace(0, -1, inplace=True)

# df['ema34_sig'] = (df['Close']>=df['ema34']).astype(np.int).replace(0, -1)

# df['wma34_sig'] = (df['Close']>=df['wma34']).astype(np.int).replace(0, -1)

# df['ADXR3_sig'] = np.sign(df['ADXR3'])
# np.unique(df['ADXR3_sig'])

# df['mom34_sig'] = np.sign(df['mom34'])
# np.unique(df['mom34_sig'])
# # mom can't be 0
# # print(np.unique(df['mom9_sig']))

# df['K_percent_sig'] = np.sign(df['K_percent']-df['K_percent'].shift(1))
# df['D_percent_sig'] = np.sign(df['D_percent']-df['D_percent'].shift(1))
# df['W_percent_sig'] = np.sign(df['W_percent']-df['W_percent'].shift(1))
# # % can't be 0
# # df['W_percent_sig'].replace(0,np.nan,inplace=True)
# # print(np.unique(df['W_percent_sig']))

# df['ADO_sig'] = np.sign(df['ADO']-df['ADO'].shift(1))
# df['ADO_sig'].replace(0,np.nan,inplace=True)

# df['macd_sig'] = np.sign(df['macd']-df['macd'].shift(1))
# df['ADO_sig'].replace(0,np.nan,inplace=True)

# rsi_high = (df['rsi34']>70)*(-1)
# rsi_low = (df['rsi34']<30)*1
# rsi_mid_up = (df['rsi34']<=70) & (df['rsi34']>=30) & (df['rsi34']>df['rsi34'].shift(1))
# rsi_mid_down = (df['rsi34']<=70) & (df['rsi34']>=30) & (df['rsi34']<=df['rsi34'].shift(1))
# rsi_mid_down = rsi_mid_down*(-1)
# df['rsi34_sig'] = rsi_high + rsi_low + rsi_mid_up + rsi_mid_down
# df['rsi34_sig'].replace(0,np.nan,inplace=True)

# cci_high = (df['cci']>200)*(-1)
# cci_low = (df['cci']<-200)*1
# cci_mid_up = (df['cci']<=200) & (df['cci']>=-200) & (df['cci']>df['cci'].shift(1))
# cci_mid_down = (df['cci']<=200) & (df['cci']>=-200) & (df['cci']<=df['cci'].shift(1))
# cci_mid_down = cci_mid_down*(-1)
# df['cci_sig'] = cci_high + cci_low + cci_mid_up + cci_mid_down
# df['cci_sig'].replace(0,np.nan,inplace=True)

# df = df.dropna()
# print(df)

# ######################## get the training and testing Trend data
# n_sample = df.shape[0]
# n_train = np.int(n_sample*0.5)
# train = df.iloc[:n_train,:]
# test = df.iloc[n_train:,:]

# ind_names = ['sma34', 'ema34','wma34','ADXR3', 'mom34', 'K_percent', 'D_percent', 'rsi34', 'macd', 'W_percent', 'cci', 'ADO']
# indTrend_names =[n+'_sig' for n in ind_names]

# train_X = train[indTrend_names]
# train_Y = np.array(train[['price_mov']])

# test_X = test[indTrend_names]
# test_Y = np.array(test[['price_mov']])


# ######################## Predicting
# ind_names = ['sma34', 'ema34','wma34','ADXR3', 'mom34', 'K_percent', 'D_percent', 'rsi34', 'macd', 'W_percent', 'cci', 'ADO']
# indTrend_names =[n+'_sig' for n in ind_names]

# test_X = test[indTrend_names]
# test_Y = np.array(test[['price_mov']])
# print(test_X.tail())
# print('↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Predict start ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓')
# start = str(r1['serverTime'])
# newTime = int(start[0:10])
# print('Now ServerTime :' + datetime.datetime.fromtimestamp(newTime).isoformat()) # server time
# print(latest_data + ' next ' + interval)

# clf = LogisticRegression()
# clf.fit(test_X, test_Y.ravel())
# sixHourLater_Y_predict = clf.predict(test_X)
# print('LogisticRegression: ' + str(sixHourLater_Y_predict[-1]))

# clf = LinearDiscriminantAnalysis()
# clf.fit(test_X, test_Y.ravel())
# sixHourLater_Y_predict = clf.predict(test_X)
# print('LinearDiscriminantAnalysis: ' + str(sixHourLater_Y_predict[-1]))
# print('################# Predict end #################')
"""
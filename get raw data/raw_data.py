import requests
import datetime

import pandas as pd
import numpy as np
import talib


################### data preprocessing
r1 = requests.get('http://api.binance.com/api/v3/time').json()
period = 86400 * 180 * 1000  # ms
startTime = r1['serverTime'] - period
raw_Data = requests.get('https://api.binance.com/api/v1/klines?symbol=ETHUSDT&interval=1d&startTime=' + str(startTime)).json()
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
for c in raw_Data:
    start = str(c[0])
    newTime = int(start[0:10])
    date = datetime.datetime.fromtimestamp(newTime).isoformat()
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
df['sma9'] = talib.SMA(price, 9)  # I1
df['ema9'] = talib.EMA(price, 9)  # I2
df['mom9'] = talib.MOM(price, 9)  # I3
df['K_percent'], df['D_percent'] = talib.STOCHF(df['High'], df['Low'], df['Close'])  # I4 I5
df['rsi6'] = talib.RSI(price, 6)  # I6
df['macd'], dif, dem = talib.MACD(price)  # I7
df['W_percent'] = talib.WILLR(df['High'], df['Low'], df['Close'])  # I8
df['cci'] = talib.CCI(df['High'], df['Low'], df['Close'])  # I9
df['ADO'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])  # I10
# print(ADO)

# price.pct_change().head()
df['price_mov'] = np.sign(price.pct_change().shift(-1))
df=df.dropna()
# print(df)


################### split data to train and test group
n_sample = df.shape[0]
n_train = np.int(n_sample*0.5)
train = df.iloc[:n_train,:]
test = df.iloc[n_train:,:]
# print(test)

# print(df.columns)
train_X = train[['sma9', 'ema9', 'mom9', 'K_percent', 'D_percent', 'rsi6', 'macd', 'W_percent', 'cci', 'ADO']]
train_Y = np.array(train[['price_mov']])

test_X = train[['sma9', 'ema9', 'mom9', 'K_percent', 'D_percent', 'rsi6', 'macd', 'W_percent', 'cci', 'ADO']]
test_Y = np.array(test[['price_mov']])


################### normalization the indicators
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(train_X)
train_X_scale = scaler.transform(train_X)
test_X_scale = scaler.transform(test_X)
# print(test_X_scale[:3])


################### build the model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_X_scale, train_Y)
train_Y_predict = clf.predict(train_X_scale)
test_Y_predict = clf.predict(test_X_scale)


################### evaluate the model
from sklearn.metrics import confusion_matrix
confusion_matrix(train_Y, train_Y_predict)
# print(confusion_matrix(train_Y, train_Y_predict))
p_true = pd.DataFrame(confusion_matrix(train_Y, train_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
p_true = pd.DataFrame(confusion_matrix(test_Y, test_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
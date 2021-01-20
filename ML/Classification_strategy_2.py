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
#     1499040000000,      // 開盤時間 [0]
#     "0.01634790",       // 開盤價格 [1]
#     "0.80000000",       // 最高價格 [2]
#     "0.01575800",       // 最低價格 [3]
#     "0.01577100",       // 收盤價格 [4]
#     "148976.11427815",  // 成交數量 [5]
#     1499644799999,      // 收盤時間 [6]
#     "2434.19055334",    // 成交金额 [7]
#     308,                // 成交筆數 [8]
#     "1756.87402397",    // taker成交數量 [9]
#     "28.46694368",      // taker成交金额 [10]
#     "17928899.62484339" // 忽略
#   ]
# ]

start_timestamp_Array = []
date_Array = []
Open_Time_Array = []
Open_Array = []
High_Array = []
Low_Array = []
Close_Array = []
Volume_Array = []
Total_Array = []
Order_Array = []
Taker_Volume_Array = []
Taker_Total_Array = []

previous_start_timestamp = 0
previous_end_timestamp = 0
for period in range(70):
    """
    print(previous_start_timestamp)
    print('    VVVVV')
    print(previous_end_timestamp)
    """
    for i, c in enumerate(raw_Data):
        start_timestamp = str(c[0])
        if i == 0:
            previous_start_timestamp = start_timestamp
        if i == len(raw_Data)-1:
            # print('Last data: ' + str(c))
            previous_end_timestamp = str(c[0])
        newTime = int(start_timestamp[0:10])
        date = datetime.datetime.fromtimestamp(newTime).isoformat()
        
        start_timestamp_Array.insert(i, int(c[0]))
        date_Array.insert(i, date)
        Open_Time_Array.insert(i, int(date[11:13])*60 + int(date[14:16]))
        Open_Array.insert(i, float(c[1]))
        High_Array.insert(i, float(c[2]))
        Low_Array.insert(i, float(c[3]))
        Close_Array.insert(i, float(c[4]))
        Volume_Array.insert(i, float(c[5]))
        Total_Array.insert(i, float(c[7]))
        Order_Array.insert(i, int(c[8]))
        Taker_Volume_Array.insert(i, float(c[9]))
        Taker_Total_Array.insert(i, float(c[10]))
        new_start_timestamp = int(previous_start_timestamp) - (int(previous_end_timestamp) - int(previous_start_timestamp))
        new_end_timestamp = int(previous_start_timestamp) - 300000
    raw_Data = requests.get('https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=' + interval + '&startTime=' + str(new_start_timestamp) + '&endTime=' + str(new_end_timestamp)).json()

df_numpy = {
    'timestamp': start_timestamp_Array,
    'date': date_Array,
    'Open_Time': Open_Time_Array,
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
# print (df)

df['sma34'] = talib.SMA(price.values, 7)  # I1
# df['sma200'] = talib.SMA(price, 200)
df['ema34'] = talib.EMA(price.values, 7)  # I2
df['wma34'] = talib.WMA(price.values, 7)
df['ADXR3'] = talib.ADXR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=3)
df['mom34'] = talib.MOM(price.values, 10)  # I3
df['K_percent'], df['D_percent'] = talib.STOCHF(df['High'].values, df['Low'].values, df['Close'].values)  # I4 I5
df['rsi34'] = talib.RSI(price.values, 10)  # I6
df['macd'], dif, dem = talib.MACD(price.values)  # I7
df['W_percent'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values)  # I8
df['cci'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values)  # I9
df['ADO'] = talib.ADOSC(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values)  # I10
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
train_X = train[['sma34', 'ema34', 'wma34','ADXR3', 'mom34', 'K_percent', 'D_percent', 'rsi34', 'macd', 'W_percent', 'cci', 'ADO']]
train_Y = np.array(train[['price_mov']])

test_X = test[['sma34', 'ema34', 'wma34','ADXR3', 'mom34', 'K_percent', 'D_percent', 'rsi34', 'macd', 'W_percent', 'cci', 'ADO']]
test_Y = np.array(test[['price_mov']])


################### normalization the indicators
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(train_X)
train_X_scale = scaler.transform(train_X)
test_X_scale = scaler.transform(test_X)
# print(test_X_scale[:3])


################### build the LogisticRegression model 
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_X_scale, train_Y.ravel())
train_Y_predict = clf.predict(train_X_scale)
test_Y_predict = clf.predict(test_X_scale)
# print(train_Y_predict)
# print(test_Y_predict)
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


################### build the SVM model 
from sklearn.svm import SVC
clf = SVC(C=100, kernel='rbf', gamma=5)
clf.fit(train_X_scale, train_Y.ravel())
train_Y_predict = clf.predict(train_X_scale)
test_Y_predict = clf.predict(test_X_scale)
print('-------------------- SVM model --------------------')
p_true = pd.DataFrame(confusion_matrix(train_Y, train_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
print()
p_true = pd.DataFrame(confusion_matrix(test_Y, test_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
print()

from sklearn.metrics import classification_report
report = classification_report(np.array(train_Y), list(train_Y_predict), output_dict=True)
print(pd.DataFrame(report))
print()
test_report = classification_report(np.array(test_Y), list(test_Y_predict), output_dict=True)
print(pd.DataFrame(test_report))



################### build the Random Forest model 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50)
clf.fit(train_X_scale, train_Y.ravel())
train_Y_predict = clf.predict(train_X_scale)
test_Y_predict = clf.predict(test_X_scale)
print('-------------------- Random Forest model --------------------')
p_true = pd.DataFrame(confusion_matrix(train_Y, train_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
print()
p_true = pd.DataFrame(confusion_matrix(test_Y, test_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
print()

from sklearn.metrics import classification_report
report = classification_report(np.array(train_Y), list(train_Y_predict), output_dict=True)
print(pd.DataFrame(report))
print()
test_report = classification_report(np.array(test_Y), list(test_Y_predict), output_dict=True)
print(pd.DataFrame(test_report))



################### build the LDA model 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(train_X_scale, train_Y.ravel())
train_Y_predict = clf.predict(train_X_scale)
test_Y_predict = clf.predict(test_X_scale)
print('-------------------- LDA model --------------------')
p_true = pd.DataFrame(confusion_matrix(train_Y, train_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
print()
p_true = pd.DataFrame(confusion_matrix(test_Y, test_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
print()

from sklearn.metrics import classification_report
report = classification_report(np.array(train_Y), list(train_Y_predict), output_dict=True)
print(pd.DataFrame(report))
print()
test_report = classification_report(np.array(test_Y), list(test_Y_predict), output_dict=True)
print(pd.DataFrame(test_report))



################### build the QDA model 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis()
clf.fit(train_X_scale, train_Y.ravel())
train_Y_predict = clf.predict(train_X_scale)
test_Y_predict = clf.predict(test_X_scale)
print('-------------------- QDA model --------------------')
p_true = pd.DataFrame(confusion_matrix(train_Y, train_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
print()
p_true = pd.DataFrame(confusion_matrix(test_Y, test_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
print(p_true)
print()

from sklearn.metrics import classification_report
report = classification_report(np.array(train_Y), list(train_Y_predict), output_dict=True)
print(pd.DataFrame(report))
print()
test_report = classification_report(np.array(test_Y), list(test_Y_predict), output_dict=True)
print(pd.DataFrame(test_report))



################### Merge all the model's results in the table
# define the table function
def get_ML_perf(clf, title='Logistic Regression'):
    perf_df = pd.DataFrame(index=['train_Accuracy','train_F_mearsure', 'test_Accuracy','test_F_mearsure'])

    clf.fit(train_X_scale, train_Y.ravel())
    train_Y_predict = clf.predict(train_X_scale)
    test_Y_predict = clf.predict(test_X_scale)
    out=[]

    ## train ##
    report = classification_report(np.array(train_Y), list(train_Y_predict), output_dict=True)
    out.append(report['macro avg']['precision'])
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    out.append(2*precision*recall/(precision+recall))
    
    ## test
    report = classification_report(np.array(test_Y), list(test_Y_predict), output_dict=True)
    out.append(report['macro avg']['precision'])
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    out.append(2*precision*recall/(precision+recall))

    perf_df[title] = out
    return(perf_df)

# run all models
print("---------------------------------------- all the model's results ----------------------------------------")
perf_all = []
## LG
clf = LogisticRegression()
perf_all.append(get_ML_perf(clf, title='Logistic Regression'))

## SVM
clf = SVC(C=50, kernel='rbf', gamma=1)
perf_all.append(get_ML_perf(clf, title='SVM'))

## Random Forest
clf = RandomForestClassifier(n_estimators=5)
perf_all.append(get_ML_perf(clf, title='Random Forest'))

## LDA
clf = LinearDiscriminantAnalysis()
perf_all.append(get_ML_perf(clf, title='LDA'))

## QDA
clf = QuadraticDiscriminantAnalysis()
perf_all.append(get_ML_perf(clf, title='QDA'))

print(pd.concat(perf_all, axis=1))





################### convert to trend prediction data
df['sma34_sig'] = (df['Close']>=df['sma34']).astype(np.int)
df['sma34_sig'].replace(0, -1, inplace=True)

# df['sma200_sig'] = (df['Close']>=df['sma200']).astype(np.int)
# df['sma200_sig'].replace(0, -1, inplace=True)

df['ema34_sig'] = (df['Close']>=df['ema34']).astype(np.int).replace(0, -1)

df['wma34_sig'] = (df['Close']>=df['wma34']).astype(np.int).replace(0, -1)

df['ADXR3_sig'] = np.sign(df['ADXR3'])
np.unique(df['ADXR3_sig'])

df['mom34_sig'] = np.sign(df['mom34'])
np.unique(df['mom34_sig'])
# mom can't be 0
# print(np.unique(df['mom9_sig']))

df['K_percent_sig'] = np.sign(df['K_percent']-df['K_percent'].shift(1))
df['D_percent_sig'] = np.sign(df['D_percent']-df['D_percent'].shift(1))
df['W_percent_sig'] = np.sign(df['W_percent']-df['W_percent'].shift(1))
# % can't be 0
# df['K_percent_sig'].replace(0,np.nan,inplace=True)
# df['D_percent_sig'].replace(0,np.nan,inplace=True)
# df['W_percent_sig'].replace(0,np.nan,inplace=True)

df['ADO_sig'] = np.sign(df['ADO']-df['ADO'].shift(1))
df['ADO_sig'].replace(0,np.nan,inplace=True)

df['macd_sig'] = np.sign(df['macd']-df['macd'].shift(1))
df['ADO_sig'].replace(0,np.nan,inplace=True)

rsi_high = (df['rsi34']>70)*(-1)
rsi_low = (df['rsi34']<30)*1
rsi_mid_up = (df['rsi34']<=70) & (df['rsi34']>=30) & (df['rsi34']>df['rsi34'].shift(1))
rsi_mid_down = (df['rsi34']<=70) & (df['rsi34']>=30) & (df['rsi34']<=df['rsi34'].shift(1))
rsi_mid_down = rsi_mid_down*(-1)
df['rsi34_sig'] = rsi_high + rsi_low + rsi_mid_up + rsi_mid_down
df['rsi34_sig'].replace(0,np.nan,inplace=True)

cci_high = (df['cci']>200)*(-1)
cci_low = (df['cci']<-200)*1
cci_mid_up = (df['cci']<=200) & (df['cci']>=-200) & (df['cci']>df['cci'].shift(1))
cci_mid_down = (df['cci']<=200) & (df['cci']>=-200) & (df['cci']<=df['cci'].shift(1))
cci_mid_down = cci_mid_down*(-1)
df['cci_sig'] = cci_high + cci_low + cci_mid_up + cci_mid_down
df['cci_sig'].replace(0,np.nan,inplace=True)

df = df.dropna()
print(df)

######################## get the training and testing Trend data
n_sample = df.shape[0]
n_train = np.int(n_sample*0.5)
train = df.iloc[:n_train,:]
test = df.iloc[n_train:,:]
PNL_test = test

ind_names = ['sma34', 'ema34','wma34','ADXR3', 'mom34', 'K_percent', 'D_percent', 'rsi34', 'macd', 'W_percent', 'cci', 'ADO']
indTrend_names =[n+'_sig' for n in ind_names]

train_X = train[indTrend_names]
train_Y = np.array(train[['price_mov']])

test_X = test[indTrend_names]
test_Y = np.array(test[['price_mov']])

################### Merge all the Trend data model's results in the table
# define the table function
def get_ML_perf(clf,title='Logistic Regression', train_X=train_X,train_Y=train_Y, test_X = test_X,test_Y=test_Y):
    perf_df = pd.DataFrame(index=['train_Accuracy','train_F_mearsure', 'test_Accuracy','test_F_mearsure'])

    clf.fit(train_X, train_Y.ravel())
    train_Y_predict = clf.predict(train_X)
    test_Y_predict = clf.predict(test_X)
    out=[]
    
    ## train ##
    report = classification_report(np.array(train_Y), list(train_Y_predict), output_dict=True)
    out.append(report['macro avg']['precision'])
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    out.append(2*precision*recall/(precision+recall))
    
    ## test
    report = classification_report(np.array(test_Y), list(test_Y_predict), output_dict=True)
    out.append(report['macro avg']['precision'])
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    out.append(2*precision*recall/(precision+recall))
    
    perf_df[title] = out
    return(perf_df)

# run all models
print("---------------------------------------- all the model's results in Trend data ----------------------------------------")
perf_all = []
## LG
clf = LogisticRegression()
perf_all.append(get_ML_perf(clf, title='Logistic Regression'))

## SVM
clf = SVC(C=50, kernel='rbf', gamma=1)
perf_all.append(get_ML_perf(clf, title='SVM'))

## Random Forest
clf = RandomForestClassifier(n_estimators=5)
perf_all.append(get_ML_perf(clf, title='Random Forest'))

## LDA
clf = LinearDiscriminantAnalysis()
perf_all.append(get_ML_perf(clf, title='LDA'))

## QDA
clf = QuadraticDiscriminantAnalysis()
perf_all.append(get_ML_perf(clf, title='QDA'))

print(pd.concat(perf_all, axis=1))


######################## Predicting
from sklearn.feature_selection import f_regression
ind_names = ['sma34', 'ema34','wma34','ADXR3', 'mom34', 'K_percent', 'D_percent', 'rsi34', 'macd', 'W_percent', 'cci', 'ADO']
indTrend_names =[n+'_sig' for n in ind_names]

test_X = test[indTrend_names]
test_Y = np.array(test[['price_mov']])
print(test_X.tail())
print('↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Predict start ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓')
start = str(r1['serverTime'])
newTime = int(start[0:10])
print('Now ServerTime :' + datetime.datetime.fromtimestamp(newTime).isoformat()) # server time
print(latest_data + ' next ' + interval)

clf = LogisticRegression()
clf.fit(test_X, test_Y.ravel())
sixHourLater_Y_predict = clf.predict(test_X)
print(sixHourLater_Y_predict)
LS_test = sixHourLater_Y_predict
# print(f_regression(test_X, test_Y.ravel())[1])
print('LogisticRegression: ' + str(sixHourLater_Y_predict[-1]))

clf = LinearDiscriminantAnalysis()
clf.fit(test_X, test_Y.ravel())
sixHourLater_Y_predict = clf.predict(test_X)
print('LinearDiscriminantAnalysis: ' + str(sixHourLater_Y_predict[-1]))
print('################# Predict end #################')


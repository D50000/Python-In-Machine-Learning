import requests
import datetime

import pandas as pd
import numpy as np
import talib


################### data preprocessing
r1 = requests.get('http://api.binance.com/api/v3/time').json()
period = 86400 * 360 * 1000  # ms
startTime = r1['serverTime'] - period
raw_Data = requests.get('https://api.binance.com/api/v1/klines?symbol=ETHUSDT&interval=1d').json()
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
df['sma34'] = talib.SMA(price, 34)  # I1
df['ema34'] = talib.EMA(price, 34)  # I2
df['mom34'] = talib.MOM(price, 34)  # I3
df['K_percent'], df['D_percent'] = talib.STOCHF(df['High'], df['Low'], df['Close'])  # I4 I5
df['rsi34'] = talib.RSI(price, 34)  # I6
df['macd'], dif, dem = talib.MACD(price)  # I7
df['W_percent'] = talib.WILLR(df['High'], df['Low'], df['Close'])  # I8
df['cci'] = talib.CCI(df['High'], df['Low'], df['Close'])  # I9
df['ADO'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])  # I10
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
train_X = train[['sma34', 'ema34', 'mom34', 'K_percent', 'D_percent', 'rsi34', 'macd', 'W_percent', 'cci', 'ADO']]
train_Y = np.array(train[['price_mov']])

test_X = test[['sma34', 'ema34', 'mom34', 'K_percent', 'D_percent', 'rsi34', 'macd', 'W_percent', 'cci', 'ADO']]
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


################### build the LinearRegression model 
# from sklearn.linear_model import LinearRegression
# clf = LinearRegression()
# clf.fit(train_X_scale, train_Y.ravel())
# train_Y_predict = clf.predict(train_X_scale)
# test_Y_predict = clf.predict(test_X_scale)
# print(train_Y_predict)
# print(test_Y_predict)
# # evaluate the model
# # T/F accuracy
# from sklearn.metrics import confusion_matrix
# confusion_matrix(train_Y, train_Y_predict)
# # print(confusion_matrix(train_Y, train_Y_predict))
# print('-------------------- LinearRegression model --------------------')
# p_true = pd.DataFrame(confusion_matrix(train_Y, train_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
# print(p_true)
# print()
# p_true = pd.DataFrame(confusion_matrix(test_Y, test_Y_predict, labels=[-1, 1]), columns=['-1_predidct', '1_predidct'], index=['-1_true', '1_true'])
# print(p_true)
# print()
# # Precision Recall
# from sklearn.metrics import classification_report
# report = classification_report(np.array(train_Y), list(train_Y_predict), output_dict=True)
# print(pd.DataFrame(report))
# print()
# test_report = classification_report(np.array(test_Y), list(test_Y_predict), output_dict=True)
# print(pd.DataFrame(test_report))


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

df['ema34_sig'] = (df['Close']>=df['ema34']).astype(np.int).replace(0, -1)

df['mom34_sig'] = np.sign(df['mom34'])
np.unique(df['mom34_sig'])
# mom can't be 0
# print(np.unique(df['mom9_sig']))

df['K_percent_sig'] = np.sign(df['K_percent']-df['K_percent'].shift(1))
df['D_percent_sig'] = np.sign(df['D_percent']-df['D_percent'].shift(1))
df['W_percent_sig'] = np.sign(df['W_percent']-df['W_percent'].shift(1))
# % can't be 0
# df['W_percent_sig'].replace(0,np.nan,inplace=True)
# print(np.unique(df['W_percent_sig']))

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

ind_names = ['sma34','ema34','mom34', 'K_percent', 'D_percent', 'rsi34', 'macd', 'W_percent', 'cci', 'ADO']
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
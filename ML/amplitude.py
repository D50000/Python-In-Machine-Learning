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
import numpy as np
import statistics 
# import talib

r1 = requests.get('http://api.binance.com/api/v3/time').json()
period = 86400 * 360 * 1000  # ms
startTime = r1['serverTime'] - period
interval = '5m'
raw_Data = requests.get('https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=' + interval).json()
# print(raw_Data)

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

dic = {
    'date': date_Array,
    'Open': Open_Array,
    'High': High_Array,
    'Low': Low_Array,
    'Close': Close_Array,
    'Volume': Volume_Array
    }
df = pd.DataFrame(data=dic)
# print(df['High'].values)
df['OCA'] = abs((df['Close'].values - df['Open'].values)) / df['Open'].values
df['HLA'] = df['High'].values / ((df['High'].values + df['Low'].values) / 2)
df=df.dropna()
print(df['OCA'])
print(df['HLA'])
print('======================================')
print('mean: %f' %(statistics.mean(df['HLA'].values)))
print('stdev: %f' %(statistics.stdev(df['HLA'].values)))
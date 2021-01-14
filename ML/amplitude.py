# ============================================================
# node-binance-api
# https://binance-docs.github.io/apidocs/spot/en/#market-data-endpoints
# ============================================================
# Copyright 2021, Eddie Hsu (D50000)
# Released MySelf
# ============================================================

import requests
import datetime

import pandas as pd
import numpy as np
import statistics 
# import talib

r1 = requests.get('https://fapi.binance.com/fapi/v1/time').json()
period = 86400 * 360 * 1000  # ms
startTime = r1['serverTime'] - period
interval = '5m'
raw_Data = requests.get('https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&limit=500&interval=' + interval).json()
# print(raw_Data)

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
        print(c)
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

dic = {
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
print('======================================')
print('Taker_Volume mean: %f' %(statistics.mean(df['Taker_Volume'].values)))
print('stdev: %f' %(statistics.stdev(df['Taker_Volume'].values)))

for index, row in df.iterrows():
    if row['Taker_Volume'] > 23000:
        print(row)
# print(df.tail(50))
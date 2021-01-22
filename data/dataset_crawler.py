# ============================================================
# node-binance-api
# https://binance-docs.github.io/apidocs/futures/en/#market-data-endpoints
# ============================================================
# Copyright 2021, Eddie Hsu (D50000)
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
for period in range(120):
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
    'date': date_Array,
    'timestamp': start_timestamp_Array,
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
# df.index = pd.to_datetime(df.date.astype(np.str))
print (df)
df.to_csv("eth_5m.csv", index_label="index")
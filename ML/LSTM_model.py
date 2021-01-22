# ============================================================
# node-binance-api

# ============================================================
# Copyright 2021, Eddie Hsu (D50000)
# Released MySelf
# ============================================================

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
#plt.show() # 有了%matplotlib inline 就可以省掉plt.show()了

def readTrain():
	train = pd.read_csv("../data/eth_5m.csv")
	return train

def normalize(train):
	train = train.drop(["index"], axis=1)
	train = train.drop(["date"], axis=1)
	train = train.drop(["timestamp"], axis=1)
	train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
	return train_norm

def buildTrain(train, pastDay=30, futureDay=5):
	X_train, Y_train = [], []
	for i in range(train.shape[0]-futureDay-pastDay):
		X_train.append(np.array(train.iloc[i:i+pastDay]))
		Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["Close"]))
	return np.array(X_train), np.array(Y_train)

def splitData(X, Y, rate):
	X_train = X[int(X.shape[0]*rate):]
	Y_train = Y[int(Y.shape[0]*rate):]
	X_val = X[:int(X.shape[0]*rate)]
	Y_val = Y[:int(Y.shape[0]*rate)]
	return X_train, Y_train, X_val, Y_val



data = readTrain()
train_norm = normalize(data)
print (train_norm)
# build Data, use last 30 days to predict next 5 days
X_train, Y_train = buildTrain(train_norm, 30, 5)

# split training data and validation data
X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)
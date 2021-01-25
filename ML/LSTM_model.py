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
	train = train.drop(["price_change"], axis=1)
	train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
	return train_norm

def buildTrain(train, pastDay=30, futureDay=5):
	X_train, Y_train = [], []
	for i in range(train.shape[0]+1-futureDay-pastDay):
		X_train.append(np.array(train.iloc[i:i+pastDay]))
		Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["price_LS"]))
	return np.array(X_train), np.array(Y_train)

def shuffle(X, Y):
	np.random.seed(10)
	randomList = np.arange(X.shape[0])
	np.random.shuffle(randomList)
	return X[randomList], Y[randomList]

def splitData(X, Y, rate):
	X_train = X[int(X.shape[0]*rate):]
	Y_train = Y[int(Y.shape[0]*rate):]
	X_val = X[:int(X.shape[0]*rate)]
	Y_val = Y[:int(Y.shape[0]*rate)]
	return X_train, Y_train, X_val, Y_val

def buildOneToOneModel(shape):
	model = Sequential()
	model.add(LSTM(10, input_length=shape[1], input_dim=shape[2],return_sequences=True))
	# output shape: (1, 1)
	model.add(TimeDistributed(Dense(1)))    # or use model.add(Dense(1))
	model.compile(loss="mse", optimizer="adam")
	print(model.summary())
	return model

def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
    # output shape: (1, 1)
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model




data = readTrain()
train_norm = normalize(data)
print (train_norm)
# build Data, use last 1 days to predict next 1 days
X_train, Y_train = buildTrain(train_norm, 12, 1)
# shuffle the data, and random seed is 10
# X_train, Y_train = shuffle(X_train, Y_train)
# split training data and validation data
X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

# from 2 dimmension to 3 dimension
# Y_train = Y_train[:,np.newaxis]
# Y_val = Y_val[:,np.newaxis]

model = buildManyToOneModel(X_train.shape)
# callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
# history = model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
history = model.fit(X_train, Y_train, epochs=500, batch_size=128, validation_data=(X_val, Y_val))
print(history.history)



# evaluation
# model.evaluate(X_train, Y_train)
print("========================================================")
results = model.evaluate(X_val, Y_val)
print(results)
# weight, Bias
# print(model.layers[0].get_weights())

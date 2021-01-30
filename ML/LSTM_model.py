# ============================================================
# node-binance-api

# ============================================================
# Copyright 2021, Eddie Hsu (D50000)
# Released MySelf
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
	# train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
	scaler = MinMaxScaler(feature_range=(0, 1))
	train_norm = scaler.fit_transform(train)
	train_norm = pd.DataFrame(train_norm, columns = ['Open_Time','Open','High','Low','Close','Volume','Total','Order','Taker_Volume','Taker_Total'])
	return train_norm, scaler

def buildTrain(train, pastDay=30, futureDay=5):
	X_train, Y_train = [], []
	for i in range(train.shape[0]+1-futureDay-pastDay):
		X_train.append(np.array(train.iloc[i:i+pastDay]))
		Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["Close"]))
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
train_norm, scaler = normalize(data)
print (train_norm)
# build Data, use last 36 days to predict next 1 days
X_train, Y_train = buildTrain(train_norm, 36, 1)
# split training data and validation data
X_train, Y_train, X_test, Y_test = splitData(X_train, Y_train, 0.1)

model = buildManyToOneModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
train_history = model.fit(X_train, Y_train, epochs=500, batch_size=128, validation_split=0.1, callbacks=[callback])


# evaluation
print("========================================================")
results = model.evaluate(X_test, Y_test)
print(results)

plt.plot(train_history.history['loss'])  
plt.plot(train_history.history['val_loss'])  
plt.title('Train History')  
plt.ylabel('loss')  
plt.xlabel('Epoch')  
plt.legend(['loss', 'val_loss'], loc='upper left')  
plt.show()

# PREDICTION
Predict_X_test = model.predict(X_test)
print(type(Predict_X_test))
print("+++++++++++++++++++++++++++++++++++++++++++++")
# Convert dataframe
# Predict_X_test = np.append(Predict_X_test, Predict_X_test, axis=1)
# Predict_X_test = pd.DataFrame(Predict_X_test, columns = ['Close'])
print(type(Predict_X_test))
print(Predict_X_test)
Y_test = pd.DataFrame(Y_test, columns = ['Close'])
# scaler.inverse
Predict_X_test = scaler.inverse_transform(Predict_X_test)
Y_test = scaler.inverse_transform(Y_test)
print(Predict_X_test.tolist())
print(Y_test.tolist())
plt.plot(Predict_X_test.tolist())  
plt.plot(Y_test.tolist())  
plt.title('Predict Y vs Y')  
plt.ylabel('time')  
plt.xlabel('price')  
plt.legend(['Predict_X_test', 'Y_test'], loc='upper left')  
plt.show()

# save the train model
# from keras.models import model_from_json
# json_string = model.to_json() with open("model.config", "w") as text_file:    
# text_file.write(json_string)
# model.save_weights("model.weight")
from keras.models import load_model
model.save('model.h5')  # creates a HDF5 file 'model.h5'
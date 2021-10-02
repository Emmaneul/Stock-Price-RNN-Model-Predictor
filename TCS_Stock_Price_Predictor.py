# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 11:23:30 2021

@author: Emmanuel Ndivhuwo Masindi
"""

#IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler 

#LOADING DATASET

training_set = pd.read_csv('Training_set - TCS_stock_history.csv')
test_set = pd.read_csv('Testing_set - TCS_stock_history.csv')

#DATA PREPROCESSING
training_set = training_set.iloc[:4435,1:2]
test_set = test_set.iloc[:,1:2]
test_set_extended = pd.concat((training_set.iloc[len(training_set.iloc[:,0])-60:,0:1], test_set), axis = 0)

training_set = training_set.values
test_set = test_set.values
test_set_extended = test_set_extended.values

sc = MinMaxScaler(feature_range=(0,1))

training_set = sc.fit_transform(training_set)
test_set_extended = sc.transform(test_set_extended)

#Prepare training set

X_train = []
y_train = []

#using step size of 60

upper_range = len(training_set)

for i in range(60, upper_range):
    X_train.append(training_set[i - 60: i, 0])
    y_train.append(training_set[i, 0])
    
    
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Prepare test set

X_test = []
y_test = []

upper_range = len(test_set_extended)

for i in range(60, upper_range):
    X_test.append(test_set_extended[i - 60: i, 0])
    y_test.append(test_set_extended[i, 0])
    
X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


#BUILDING THE MODEL

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(units = 1))


#COMPILING THE MODEL

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 10, batch_size = 32)

#ASSESSING THE MODEL'S PERFORMANCE

plt.style.use('fivethirtyeight')
plt.plot(pd.DataFrame(model.history.history['loss']))
plt.title('Model loss')
plt.ylabel('Loss')


#MAKING PREDICTIONS

predicted_stock_price = model.predict(X_test)
real_stock_price = y_test.reshape(-1,1)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price = sc.inverse_transform(real_stock_price)

#PLOTTING VISUALIZATION

plt.plot(real_stock_price, color = 'red', label = 'Real TCS Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted TCS Stock Price Trend')
plt.title('Stock Price')
plt.xlabel('Time')
plt.ylabel('TCS Stock Price')
plt.legend()
plt.show()





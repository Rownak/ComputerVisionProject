import pandas as pd
from random import random

'''
This program is to test LSTM model on a vector. the output is not good for short sample data.
'''

flow = (list(range(1,10,1)) + list(range(10,1,-1)))*100
pdata = pd.DataFrame({"a":flow, "b":flow})
pdata.b = pdata.b.shift(9)
data = pdata.iloc[10:] * random()  # some noise

import numpy as np

def _load_data(data, n_prev = 100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


def sampleData():
    
    X_train = np.arange(-100,100,0.01)
    
    X_train = X_train.reshape(-1,1,5)
    
    y_train = (X_train*2).reshape(-1,5)
    
    X_test = np.arange(-1, 1, 0.01).reshape(-1,1,5)

    y_test = (X_test*2).reshape(-1,5)
    
    return (X_train, y_train), (X_test, y_test)

def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

in_out_neurons = 10
hidden_neurons = 500

model = Sequential()
model.add(LSTM(hidden_neurons,  return_sequences=True))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")


(X_train, y_train), (X_test, y_test) = sampleData()

#(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data


model.fit(X_train, y_train, batch_size=700, nb_epoch=10, validation_split=0.05)

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

# and maybe plot it
pd.DataFrame(predicted).to_csv("predicted.csv")
pd.DataFrame(y_test).to_csv("test_data.csv")
pd.DataFrame(X_train.reshape(-1,5)).to_csv("X_train.csv")
pd.DataFrame(y_train).to_csv("y_train.csv")

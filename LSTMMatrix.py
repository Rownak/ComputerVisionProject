#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 09:59:44 2018

@author: afarhan
"""

import pandas as pd
from random import random
from PIL import Image
from matplotlib.pylab import *
import time
import numpy as np
import json
import os
import pandas as pd
import nltk
import gensim
from collections import defaultdict
from gensim import corpora, models, similarities
import cv2
from sklearn.neural_network import MLPClassifier


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

def videoData():
    a=8;


def sampleData():
    
    X_train = np.arange(-1,1,0.002)
    
    X_train = X_train.reshape(-1,1,5)
    
    y_train = (X_train*2).reshape(-1,5)
    
    X_test = np.arange(-.1, .1, 0.002).reshape(-1,1,5)

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



rowS=70
rowE=295
colS=10
colE=460

imgR = 45
imgC = 90
patternR = 9
patternC = 9
windowsize_r = 9
windowsize_c = 9
    
uniqueHist = defaultdict(list)
stepSize = 9
classLabelList = []

cap = cv2.VideoCapture('physarum.mp4')
#cap = cv2.VideoCapture(0)
start = time.time()
count=0

totalFrame = 100
frameDist = 5

X_train = []
y_train = []
X_test = []
y_test = []
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    
    if count >totalFrame+frameDist-1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
    if count%1==0 and count < totalFrame:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray[rowS:rowE,colS:colE]
        height , width =  crop.shape
        new_h=height//5
        new_w=width//5
        resize = cv2.resize(crop, (new_w, new_h))
        cv2.imshow('frame',resize)
        resize = resize/255.0
        if count == 0:
            print('xtrain ', count)
            X_train = resize.flatten().reshape(1,1,-1)
        elif count>0 and count< totalFrame-frameDist :
            X_train = np.vstack((X_train, resize.flatten().reshape(1,1,-1)))
        if count==frameDist:
            print('xtrain ', count)
            y_train = resize.flatten();
        elif count>frameDist:
           y_train = np.vstack((y_train, resize.flatten()))
           if count == totalFrame-frameDist:
               print('xt:',count)
               X_test = resize.flatten().reshape(1,1,-1)
           elif count> totalFrame-frameDist:
               print('xt:',count)
               X_test = np.vstack((X_test, resize.flatten().reshape(1,1,-1)))
        #print(np.max(frame),np.min(frame))
    if count == totalFrame:
        print('yt:',count)
        y_test = resize.flatten()
        
    elif count > totalFrame:
        print('yt:',count)
        y_test = np.vstack((y_test, resize.flatten()))
    
    count+=1


elapsed_time = time.time()-start
print('Capture speed: {0:.2f} frames per second'.format(count/elapsed_time))   
print('Total Img: ', count/5)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


in_out_neurons = 4050
hidden_neurons = 500

model = Sequential()
model.add(LSTM(hidden_neurons, input_dim=in_out_neurons, return_sequences=False))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")


#(X_train, y_train), (X_test, y_test) = sampleData()

#(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data


model.fit(X_train, y_train, batch_size=700, nb_epoch=20, validation_split=0.05)

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print('Error Rate:', np.mean(rmse))
# and maybe plot it
pd.DataFrame(predicted).to_csv("predicted.csv")
pd.DataFrame(y_test).to_csv("test_data.csv")
pd.DataFrame(X_train.reshape(-1,5)).to_csv("X_train.csv")
pd.DataFrame(y_train).to_csv("y_train.csv")

'''
p=predicted.reshape(200,45,90)
formatted = (X_train[4799].reshape(45,90) * 255 / np.max(X_train[4799])).astype('uint8')
img = Image.fromarray(formatted)
img.show()
img.save('p2.jpg')
'''
#import keras

import numpy as np
from typing import Tuple


import json
import pickle
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import StandardScaler,power_transform, MinMaxScaler
from sklearn.impute import  SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PowerTransformer

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Dropout, GlobalAveragePooling1D, Input, Concatenate, Flatten, Embedding, Reshape, Conv1D, TimeDistributed, BatchNormalization, GaussianNoise
from tensorflow.keras.models import Sequential
from numpy.random import seed
from tensorflow.random import set_seed

from tensorflow import keras
from tensorflow.keras import preprocessing


df = pd.read_csv('wind.csv')
df['Epoch'] = pd.to_datetime(df['Epoch'])
test_cutoff_date = df['Epoch'].max() - timedelta(days=59)
val_cutoff_date = test_cutoff_date - timedelta(days=14)

df_test = df[df['Epoch'] > test_cutoff_date]
df_val = df[(df['Epoch'] > val_cutoff_date) & (df['Epoch'] <= test_cutoff_date)]
df_train = df[df['Epoch'] <= val_cutoff_date]

trainX = []
trainY = []
trainY = df_train['sigmaPeak_doy'].to_numpy().reshape(-1,1)
trainX = df_train.to_numpy().reshape(-1,4,1919121)
trainX, trainY = np.array(trainX), np.array(trainY)
print(trainX.shape)
print(trainY.shape)
verbose, epochs, batch_size = 2, 70, 128

model = Sequential()
model.add(Bidirectional(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=True)))
model.add(Bidirectional(LSTM(32, activation='relu', return_sequences=True)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(trainY.shape[1]))
"""model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs))"""
trainX = np.asarray(trainX).astype('float')
trainY = np.asarray(trainY).astype('float')

model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)

model.summary()

model.save('.\\model_lstm')

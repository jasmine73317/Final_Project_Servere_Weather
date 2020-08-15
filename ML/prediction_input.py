from tensorflow import keras

import os
import datetime

import pymongo

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import lstm_prediction_functions_test as lstm

year = input('year: ')
month = input('month: ')
day = input('day: ')
city = input('city: ')

predictions, DateandTime = lstm.lstm_predict(city, year, month, day)
label_col_index = input('column to plot: (Humidity, Pressure, Temperature, Wind_Speed)')

plt.scatter(DateandTime, predictions[label_col_index],
                  marker = 'X', edgecolors = 'k', label = 'Predictions',
                  c = '#ff7f0e', s=64)
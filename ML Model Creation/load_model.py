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

conn = 'mongodb+srv://newuser:datasciproj2@cluster0.iadrt.mongodb.net/test'
client = pymongo.MongoClient(conn)
db = client.Weather
collection = db["LSTMRNN"]

model = keras.models.load_model('../MODELS/Denver_model.h5')
print(model.summary())



results = collection.find({
    'Year':'2012',
    'City': 'Denver',
    'Month': '08',
    'Day': '05'
})
for result in results:
    print(result)
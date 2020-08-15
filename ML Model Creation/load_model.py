from tensorflow import keras

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

model = keras.models.load_model('../MODELS/Denver_model.h5')
print(model.summary())

predict_df = train_df.tail(24) #any dataframe 24 rows and 11 columns 
prediction_set = predict_df.to_numpy()[np.newaxis,:]

predictions = model.predict(prediction_set)

predictions = pd.DataFrame(predictions[0])

#replace columns with original columns
i = 0
for column in train_std.index:
    predictions.rename(columns = {i: column}, inplace = True)
    i+=1
    
predictions = (predictions * train_std) + train_mean

predictions.head()
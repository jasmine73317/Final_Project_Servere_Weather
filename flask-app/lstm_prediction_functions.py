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

def lstm_predict(City, Year, Month, Day):
    numeric_features = [
        'Humidity',
        'Pressure',
        'Temperature',
        'Wind_Speed',
        #     'Year Temp Max', 
        #     'Year Temp Min', 
        'Year Temp Average',
        #     'Month Temp Max', 
        #     'Month Temp Min', 
        'Month Temp Average',
        #     'Year Pressure Max', 
        #     'Year Pressure Min', 
        'Year Pressure Average',
        #     'Month Pressure Max', 
        #     'Month Pressure Min', 
        'Month Pressure Average',
        #     'Year Wind_Speed Max', 
        #     'Year Wind_Speed Min', 
        'Year Wind_Speed Average',
        #     'Month Wind_Speed Max', 
        #     'Month Wind_Speed Min',
        'Month Wind_Speed Average', 
        #     'Year Humidity Max', 
        #     'Year Humidity Min',
        'Year Humidity Average', 
        #     'Month Humidity Max', 
        #     'Month Humidity Min',
        'Month Humidity Average'
    ]

    conn = 'mongodb+srv://newuser:datasciproj2@cluster0.iadrt.mongodb.net/test'
    client = pymongo.MongoClient(conn)
    db = client.Weather
    collection = db["LSTMRNN"]

    results = collection.find({
        'City': City,
        'Year': Year,
        'Month': Month,
        'Day': Day
    })
    csv = []
    for result in results:
        csv.append(result)
    csv = pd.DataFrame(csv)
    main_cols = ['Humidity','Temperature','Wind_Speed','Pressure']
    model = keras.models.load_model(f'../MODELS/{city}_model.h5')
    # print(model.summary())

    date_time = pd.to_datetime(csv.pop('Date and Time'))
    csv.index = date_time

    csv = csv[numeric_features]
    for feature in numeric_features:
        pd.to_numeric(csv[feature])

    csv_mean = csv.mean()
    csv_std = csv.std()

    csv = (csv - csv_mean) / csv_std
        
    dates_to_find = []
    for i in range(24):
            if i < 10:
                hour = '0' + str(i)
            else:
                hour = str(i)
            
            day = str(int(Day))

            dates_to_find.append(f'{Year}-{Month}-{day} {hour}:00:00')
    dates_to_find = pd.to_datetime(pd.Series(dates_to_find))

    csv = csv.loc[f'{Year}-{Month}-{Day} 00:00:00':f'{Year}-{Month}-{day} 00:00:00']


    # predict_df = csv.between_time() #any dataframe 24 rows and 11 columns 
    prediction_set = csv.to_numpy()[np.newaxis,:]
    predictions = model.predict(prediction_set)

    predictions = pd.DataFrame(predictions[0])


    #replace columns with original columns
    i = 0
    for column in train_std.index:
        predictions.rename(columns = {i: column}, inplace = True)
        i+=1
        

    predictions = (predictions * csv_std) + csv_mean
    predictions.index = dates_to_find


    return predictions.to_dict()[0]
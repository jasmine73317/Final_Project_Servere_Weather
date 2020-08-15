# app.py    
from flask import Flask, request, jsonify, render_template
from joblib import load

from tensorflow import keras
# from keras.models import load_model
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf

model_CHICAGO = keras.models.load_model('../models/binary_chicagonew_model.h5')
CHICAGO_trans = load("../models/Chicagonew_labeltransfomer.joblib")
CHICAGO_Xscaler = load("../models/Chicagonew_Xscaler.joblib")

app = Flask(__name__)



# type in 'index' at the end of url to see sample test *** note flask needs a templates folder 
@app.route('/')
def index():
    return render_template('index.html')

# A welcome message to test our server
@app.route('/summary/')
def summary():
    
     return render_template('generic.html') 

@app.route('/models/')
def models():
    
     return render_template('models.html') 

@app.route('/city_weather/')
def city_weather():
    
     return render_template('city_weather.html') 





@app.route('/weatherpredict',methods=["GET","POST"])
def weatherpredict():
    # print("......")
    prediction_value=  None
    if request.method == "GET":
        return render_template('weatherpredict.html', prediction_value= prediction_value)


    elif request.method =="POST":
        Temperature = float(request.form.get('Temperature',0))
        Humidity =float(request.form.get('Humidity',0))
        Wind_Speed = float(request.form.get('Wind_Speed',0))
        Pressure = float(request.form.get('Pressure',0))
        Month_Temp_Max = float(request.form.get('Month Temp Max',0))
        Wind_Direction = float(request.form.get('Wind_Direction',0))
        Month_Wind_Speed_Max= float(request.form.get('Month_Wind_Speed_Max',0))
        data_arr = pd.DataFrame([{'Temperature': Temperature,'Humidity':Humidity, 'Wind_Speed': Wind_Speed,'Pressure': Pressure,
        'Month Temp Max':Month_Temp_Max,'Wind_Direction':Wind_Direction,'Month Wind_Speed Max':Month_Wind_Speed_Max}])
        print(Temperature)

    
        prediction_ = model_CHICAGO.predict([CHICAGO_Xscaler.transform(data_arr)])[0]
        chosen_ = np.argmax(prediction_)
        prediction_value = CHICAGO_trans.inverse_transform([chosen_])[0]
    
        return render_template('weatherpredict.html',prediction_value= prediction_value)

    else:
        return render_template('weatherpredict.html',prediction_value = "ERROR")
  

if __name__ == '__main__':
    
    app.run(debug=True, port=5001) 








 #     df[['Temperature', 'Humidity', 'Wind_Speed',
    #    'Wind_Direction', 'Pressure','Month Temp Max','Month Wind_Speed Max']]
#                         

 # predictions = model_CHICAGO.predict([mytest_data_scaled])[0]
    # prediction_value = label_encoder.inverse_transform([np.argmax(prediction)])
    # prediction = model.predict(mytest_scaled[0:1])[0]
    # chosen = np.argmax(prediction)
    # render_template()

 # if request.method =="POST":
    #     data_arr = [request.form.get("Temperature"),("Humidity"),("Wind_Speed"),("Wind_Direction"),
    #           ("Presssure"),("Year"),("Month"),("Day"),("Hour"),
    #          ("Year Temp Max"),("Year Temp Min"),("Year Temp Average"),
    #         ("Month Temp Max"),("Month Temp Min"),("Month Temp Average"),
    #         ("Year Pressure Max"),("Year Pressure Min"),("Year Pressure Average"),
    #         ("Month Pressure Max"),("Month Pressure Min"),("Month Pressure Average"),
    #         ("Year Wind_Speed Max"),("Year Wind_Speed Min"),("Year Wind_Speed Average"),
    #         ("Month Wind_Speed Max"),("Month Wind_Speed Min"),("Month Wind_Speed Average")
    #         ("Year Humidity Max"),("Year Humidity Min"),("Year Humidity Average"),
    #         ("Month Humidity Max"),("Month Humidity Min"),("Month Humidity Average")]
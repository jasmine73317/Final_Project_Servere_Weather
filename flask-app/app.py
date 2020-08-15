# app.py    
from flask import Flask, request, jsonify, render_template
from flask import request 
import os
import lstm_prediction_functions as lstm

app = Flask(__name__)

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

# type in 'index' at the end of url to see sample test *** note flask needs a templates folder 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lstmpredict', methods = ['GET','POST'])
def weatherpredict():
    #print('.....')
    prediction_value = None
    if request.method == 'GET':
        return render_template('weatherpredict.html')
    
    elif request.method == "POST":
        year = request.form.get('Year')
        month = request.form.get('Month')
        day = request.form.get('Day')
        city = request.form.get('City')

        prediction_ = lstm.lstm_predict(city, year, month, day)

        return render_template('lstm_predict.html',predictions = prediction_)



#model input(?)
@app.route('/resource/', methods = ['POST'])
def update_text():
    data= request.data
    print('hell0')
    return data

# A welcome message to test our server
@app.route('/summary/')
def summary():
     return render_template('generic.html') 

if __name__ == '__main__':
    
    app.run(debug=True, port=5000) 

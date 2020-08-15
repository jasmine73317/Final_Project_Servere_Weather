# app.py    
from flask import Flask, request, jsonify, render_template
from joblib import load

app = Flask(__name__)

# model1 = load("flask-app/models/Humidity_Regressions.ipynb") file too large
# model2 = load("flask-app/models/Humidity_Trees.joblib") file too large

# type in 'index' at the end of url to see sample test *** note flask needs a templates folder 
@app.route('/')
def index():
    return render_template('index.html')

# second page setup
@app.route('/summary/')
def summary():
     return render_template('generic.html') 

if __name__ == '__main__':
    # print(model1, model2)
    app.run(debug=True, port=5000) 
    
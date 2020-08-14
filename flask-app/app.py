# app.py    
from flask import Flask, request, jsonify, render_template
from flask import request 

app = Flask(__name__)


# type in 'index' at the end of url to see sample test *** note flask needs a templates folder 
@app.route('/')
def index():
    return render_template('index.html')

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

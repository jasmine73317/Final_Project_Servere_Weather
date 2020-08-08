# app.py    
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# A welcome message to test our server
@app.route('/')
def hello():
    return "<h1>Welcome to our server!</h1>"

# type in 'index' at the end of url to see sample test *** note flask needs a templates folder 
@app.route('/index/')
def index(name=None):
    return render_template('test.html', name=name)

if __name__ == '__main__':
    
    app.run(debug=True, port=5000) 

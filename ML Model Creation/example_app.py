@app.route('/weatherpredict', methods = ['GET','POST'])
def weatherpredict():
    #print('.......')
    prediction_value = None
    if request.method == 'GET':

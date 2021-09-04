import numpy as np
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
from tensorflow.keras.models import load_model
# Loading model to compare the results
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [int_features]
    print("Features are ", final_features, " No of features =", len(final_features))
    #[[15200,14300]]
    prediction = model.predict(final_features)

    output = np.round(prediction[0], 2)
    # output =10
    return render_template('index.html', prediction_text='Inflow for Tomorrow is $ {}'.format(output))


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)
    # app.run(host="127.0.0.1",port=8080)
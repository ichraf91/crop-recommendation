from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sklearn
import os
import pickle
import warnings
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.before_request
def before_request():
    if request.method == 'POST' and request.headers['Content-Type'] != 'application/json':
        return jsonify(message='Invalid Content-Type'), 400
loaded_model = pickle.load(open("model.pkl", 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
   ## N = int(request.form['Nitrogen'])
    ##P = int(request.form['Phosporus'])
   ## K = int(request.form['Potassium'])
    ##temp = float(request.form['Temperature'])
    ##humidity = float(request.form['Humidity'])
    ##ph = float(request.form['pH'])
    ##rainfall = float(request.form['Rainfall'])
    data = request.get_json()
    N = int(data['Nitrogen'])
    P = int(data['Phosporus'])
    K = int(data['Potassium'])
    temp = float(data['Temperature'])
    humidity = float(data['Humidity'])
    ph = float(data['pH'])
    rainfall = float(data['Rainfall'])


    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = loaded_model.predict(single_pred)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    ##return render_template('home.html', prediction=result)
    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)

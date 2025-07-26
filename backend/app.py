from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend')

# Load trained model
model = joblib.load("wildfire_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')  # loads index.html from ../frontend

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([
        data['X'], data['Y'], data['FFMC'], data['DMC'],
        data['DC'], data['ISI'], data['temp'], data['RH'],
        data['wind'], data['rain']
    ]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({'fire_risk': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)

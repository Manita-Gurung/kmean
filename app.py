import pickle
import numpy as np
from flask import Flask, request, render_template

# Load simple model and scaler
model = pickle.load(open("simple_model.pkl", "rb"))
scaler = pickle.load(open("simple_scaler.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [
        float(request.form['price']),
        float(request.form['service_fee']),
        float(request.form['minimum_nights']),
        float(request.form['number_of_reviews']),
        float(request.form['reviews_per_month']),
        float(request.form['calculated_host_listings_count']),
        float(request.form['availability_365'])
    ]

    scaled_features = scaler.transform([input_features])
    prediction = model.predict(scaled_features)[0]

    return render_template('index.html', prediction_text=f'The listing belongs to Cluster: {prediction}')

import os

port = int(os.environ.get("PORT", 10000))  # use Render's port or fallback to 10000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)

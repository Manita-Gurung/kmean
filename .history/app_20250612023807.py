from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load all 4 models and scalers
models = {
    "Model 1": (pickle.load(open("model1.pkl", "rb")), pickle.load(open("scaler_model1.pkl", "rb"))),
    "Model 2": (pickle.load(open("model2.pkl", "rb")), pickle.load(open("scaler_model2.pkl", "rb"))),
    "Model 3": (pickle.load(open("model3.pkl", "rb")), pickle.load(open("scaler_model3.pkl", "rb"))),
    "Model 4": (pickle.load(open("model4.pkl", "rb")), pickle.load(open("scaler_model4.pkl", "rb")))
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form['model']
    model, scaler = models[model_choice]

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
    prediction = model.predict(scaled_features)

    return render_template('index.html', prediction_text=f"{model_choice} predicts: Cluster {int(prediction[0])}")

if __name__ == '__main__':
    app.run(debug=True)

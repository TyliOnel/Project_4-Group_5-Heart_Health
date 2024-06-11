from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, static_url_path='/static')

# Load the saved Random Forest model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Extract features from the data
        features = [
            float(data['feature1']),
            float(data['feature2']),
            float(data['feature3']),
            float(data['feature4'])
        ]
        
        # Preprocess the input features
        features_scaled = scaler.transform([features])
        
        # Predict using the model
        prediction = model.predict(features_scaled)[0]
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)
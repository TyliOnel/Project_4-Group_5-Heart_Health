from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the random forest model
model = joblib.load('rf_model.pkl')  

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    # Extract features from the data
    features = [data['feature1'], data['feature2'], data['feature3'], data['feature4']]
    # Predict using the model
    prediction = model.predict([features])
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
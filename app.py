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

@app.route('/model.html')
def test():
    return (
        f"This is the model<br/>"
        f"Model: {type(model).__name__}<br/>"
        f"number: {len(model.estimators_)}<br/>"
        f"features: {model.n_features_in_}<br/>"
        f"classes: {model.classes_}<br/>"
        f"parameters: {model.get_params()}<br/>"
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    print(data)
    # Extract features from the data
    features = [data['feature1'], data['feature2'], data['feature3'], data['feature4']]
    # Predict using the model
    prediction = model.predict([features])
    # Return the prediction as a JSON response
    if int(prediction[0]) == 1:
        return jsonify({'prediction': "At risk"})
    
    if int(prediction[0]) == 0:
        return jsonify({'prediction': "No risk"})

    else: 
        return jsonify({'prediction': "Unknown"})
    


if __name__ == '__main__':
    app.run(debug=True)
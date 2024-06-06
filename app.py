from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle the prediction logic here
    return "Prediction result"

if __name__ == '__main__':
    app.run(debug=True)

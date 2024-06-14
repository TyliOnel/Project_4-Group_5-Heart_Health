# Coronary Heart Disease Risk Prediction

[![Python 3.x](https://img.shields.io/badge/python-3.x-red.svg)](https://www.python.org/)
[![Pandas 2.1.4](https://img.shields.io/badge/pandas-2.1.4-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib 3.8.0](https://img.shields.io/badge/matplotlib-3.8.0-83AB57.svg)](https://matplotlib.org/)
[![Scikit-learn 1.2.2](https://img.shields.io/badge/scikit--learn-1.2.2-pink.svg)](https://scikit-learn.org/)
[![SQLAlchemy 2.0.21](https://img.shields.io/badge/sqlalchemy-2.0.21-purple.svg)](https://www.sqlalchemy.org/)
[![Plotly 5.9.0](https://img.shields.io/badge/plotly-5.9.0-yellow.svg)](https://plotly.com/)
[![Jupyter Notebook 1.0.0](https://img.shields.io/badge/jupyter--notebook-1.0.0-teal.svg)](https://jupyter.org/)
[![Seaborn 0.12.2](https://img.shields.io/badge/seaborn-0.12.2-orange.svg)](https://seaborn.pydata.org/)
[![Numpy 1.26.3](https://img.shields.io/badge/numpy-1.26.3-skyblue.svg)](https://numpy.org/)
[![TensorFlow 2.16.1](https://img.shields.io/badge/tensorflow-2.16.1-gold.svg)](https://www.tensorflow.org/)
[![Keras 3.3.3](https://img.shields.io/badge/keras-3.3.3-B95FAB.svg)](https://keras.io/)
[![Flask 2.2.5](https://img.shields.io/badge/flask-2.2.5-7CC198.svg)](https://flask.palletsprojects.com/)
[![SciPy 1.11.4](https://img.shields.io/badge/scipy-1.11.4-E48A36.svg)](https://www.scipy.org/)
[![imblearn 0.11.0](https://img.shields.io/badge/imblearn-0.11.0-brightgreen)](https://pypi.org/project/imbalanced-learn/0.11.0/)
[![statsmodels 0.14.0](https://img.shields.io/badge/statsmodels-0.14.0-brown)](https://pypi.org/project/statsmodels/0.14.0/)
[![joblib 1.2.0](https://img.shields.io/badge/joblib-1.2.0-grey)](https://pypi.org/project/joblib/1.2.0/)


## Project Overview
Our project aims to address the issue of coronary heart disease (CHD) risk prediction by leveraging machine learning techniques. Utilizing technologies like Scikit-learn and other machine learning libraries, we will build a predictive model powered by a dataset with over 3000 records. This project will incorporate several key technologies, including Python Pandas, Python Matplotlib, JavaScript Plotly, and potentially a SQL or MongoDB database.

### Objective
To predict the risk of coronary heart disease (CHD) based on various health and demographic factors.

### Key Questions to Answer
1. What are the key factors contributing to heart disease risk?
2. How accurately can we predict heart disease risk using machine learning models?

### Data Source
We will use the Kaggle dataset on heart disease for this project: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/mahdifaour/heart-disease-dataset)

### Goal
Our goal is to create an interactive website that healthcare professionals can use to input different demographic factors and utilize our model to predict a client’s risk for CHD.

### Target Variable
The target variable we aim to predict is CHDRisk (Coronary Heart Disease Risk).


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Install the following softwares with versions equal to or higher than

```
Python 3.x
Pandas 2.1.4
Matplotlib 3.8.0
Scikit-learn 1.2.2
sqlalchemy 2.0.21
Plotly 5.9.0
Jupyter Notebook 1.0.0 
Seaborn 0.12.2
Numpy 1.26.3
Tensorflow 2.16.1
Keras 3.3.3 
Flask 2.2.5
Scipy 1.11.4
imblearn 0.11.0
statsmodels  0.14.0
joblib 1.2.0
```

### Installing

Here are the installation instructions for the specified libraries and tools:

1. **Python 3.x**
   - Install Python from the official website: [Python Downloads](https://www.python.org/)

2. **Pandas 2.1.4**
   - Install Pandas using pip: 
     ```
     pip install pandas==2.1.4
     ```

3. **Matplotlib 3.8.0**
   - Install Matplotlib using pip: 
     ```
     pip install matplotlib==3.8.0
     ```

4. **Scikit-learn 1.2.2**
   - Install Scikit-learn using pip: 
     ```
     pip install scikit-learn==1.2.2
     ```

5. **SQLAlchemy 2.0.21**
   - Install SQLAlchemy using pip: 
     ```
     pip install SQLAlchemy==2.0.21
     ```

6. **Plotly 5.9.0**
   - Install Plotly using pip: 
     ```
     pip install plotly==5.9.0
     ```

7. **Jupyter Notebook 1.0.0**
   - Install Jupyter Notebook using pip: 
     ```
     pip install notebook==1.0.0
     ```

8. **Seaborn 0.12.2**
   - Install Seaborn using pip: 
     ```
     pip install seaborn==0.12.2
     ```

9. **Numpy 1.26.3**
   - Install Numpy using pip: 
     ```
     pip install numpy==1.26.3
     ```

10. **TensorFlow 2.16.1**
    - Install TensorFlow using pip: 
      ```
      pip install tensorflow==2.16.1
      ```

11. **Keras 3.3.3**
    - Install Keras using pip: 
      ```
      pip install keras==3.3.3
      ```

12. **Flask 2.2.5**
    - Install Flask using pip: 
      ```
      pip install Flask==2.2.5
      ```

13. **SciPy 1.11.4**
    - Install SciPy using pip:
      ```
      pip install scipy==1.11.4
      ```

14. **imblearn 0.11.0**
    - Install imbalanced-learn using pip:
      ```
      pip install imbalanced-learn==0.11.0
      ```

15. **statsmodels 0.14.0**
    - Install statsmodels using pip:
      ```
      pip install statsmodels==0.14.0
      ```

16. **joblib 1.2.0**
    - Install joblib using pip:
      ```
      pip install joblib==1.2.0
      ``` 

These installation rules provide clear instructions on how to install each package using pip with the specified version.

These commands can be executed in your command line interface or terminal to install the respective libraries and tools. Make sure to replace `==` with `>=` if you are open to installing later versions.

## Overview
This project aims to predict the risk of Coronary Heart Disease (CHD) using machine learning models. The dataset includes various features such as age, cholesterol level, blood pressure, and exercise level. The primary goal is to develop a reliable model that can accurately identify individuals at risk of developing CHD.

## Project Steps

### 1. Data Preparation and Cleaning
- **Loading Dataset**: The dataset was loaded from a CSV file.
- **Mapping Binary Columns**: Binary columns (e.g., 'yes/no', 'female/male') were mapped to numeric values.
- **Handling Missing Values**: Rows with missing values were dropped.
- **Class Imbalance**: The proportion of classes within the target variable was explored, revealing a notable class imbalance.

### 2. Exploratory Data Analysis (EDA)
- **Class Distribution**: The distribution of the target variable (CHDRisk) was visualized using a pie chart.
- **Normality Test**: The Shapiro-Wilk test indicated that the target variable is not normally distributed.
- **Autocorrelation Analysis**: The target variable was checked for autocorrelation and was found to be identically distributed.

### 3. Feature Analysis
- **Discrete Feature Distribution**: The frequency of discrete features by target variable class was explored.
- **Continuous Feature Distribution**: The distributions of continuous variables for both majority and minority classes in the target variable were analyzed.
- **Correlation Matrix**: A heatmap was created to visualize the correlation between features.
- **Multicollinearity Detection**: Variance Inflation Factor (VIF) was calculated to detect multicollinearity among features.

### 4. Feature Engineering
- **Mean Arterial Blood Pressure (MAP)**: Systolic and diastolic blood pressure were combined into a single indicator (MAP) to reduce multicollinearity.

### 5. Model Development and Evaluation
- **Baseline Models**: Logistic regression, neural network, and random forest models were developed and evaluated.
  - **Random Forest Model**: Achieved the highest accuracy but had a low recall score for the minority class.
  - **Logistic Regression Model**: Showed the best recall score for the minority class but had lower overall accuracy.
  - **Neural Network Model**: Demonstrated moderate performance with an accuracy of 0.71 and a recall of 0.36.

### 6. Model Optimization
- **Dropping Features**: Based on low PCA loadings, high correlation, and low feature importance.
- **Feature Engineering**: Combining blood pressure measurements into MAP.
- **Results**: The optimized random forest model (dropping features of low importance) exhibited the best performance with high accuracy and consistency across k-fold validation.

### 7. Model Deployment
A Flask web application was developed to allow users to input features and get predictions from the trained random forest model.

## Results

- **Best Model**: Random Forest with optimized features.
  - **Accuracy**: 0.82
  - **Recall**: 0.26
  - **Cross-Validation**: Consistent accuracy and recall across k-fold validation, indicating the model is not overfit.

## Conclusion
While the optimized random forest model achieved high accuracy, the recall scores for the minority class (those at risk of developing CHD) were consistently low, highlighting a significant issue of high false negatives. This underscores the need for more data collection, especially for the at-risk population, to improve the model's ability to correctly identify individuals at risk of developing CHD.

## Flask App Usage
The Flask app allows users to predict the risk of CHD by inputting features such as age, cholesterol level, blood pressure, and exercise level. The app uses the trained random forest model to provide predictions.

## Future Work
- **Data Collection**: Gathering more data, especially for the minority class, to improve recall scores.
- **Model Improvements**: Exploring advanced techniques to enhance model performance, particularly for recall of the minority class.



### Deployment of Heart Health Model

**Objective**: Deployed a machine learning application to predict heart health using a Random Forest classification model, integrating Python, CSS, JSON, Flask, and HTML.

#### Steps for Deployment

1. **Set Up the Server Environment**
   - **Choose a Hosting Platform**: Google Cloud Platform.
   - **Configure the Server**: Set up an environment compatible with Python and Flask.
   - **Install Dependencies**: Used a `requirements.txt` file to manage Python dependencies.

2. **Prepare the Codebase**
   - **Project Structure**:
     ```
    /project-root
├── app.py                   # Main Flask application file
├── data_exploration.ipynb   # Jupyter notebook for data exploration
├── rf_model.pkl             # Serialized Random Forest model
├── resources
│   └── cleaned_data.csv     # Cleaned data for the model
├── static
│   ├── images
│   │   └── heart.png        # Image used in the web pages
│   ├── script.js            # JavaScript file for client-side logic
│   └── style.css            # CSS file for styling
└── templates
    ├── index.html           # Main page template
    ├── landing.html         # Landing page template
    └── model.html           # Model details page template


     ```
   - **Model Persistence**: Save the trained Random Forest model using `pickle` or `joblib`.
   - **Create Routes**: Define Flask routes for predictions, HTML rendering, and serving static files.

3. **Develop the Flask Application**
   
     ```
   - **Create Templates**: HTML files for the frontend (`landing.html` for main page `index.html` for the analysis page, `model_info.html` for model details).
   - **Static Files**: Include CSS and images in the `static` directory.


4. **Set Up Environment Variables**
   - For Flask, set `FLASK_APP` if needed:
     ```sh
     export FLASK_APP=app.py
     ```

6. **Develop Prediction Workflow**

Connect Flask with HTML:
Data Analysis Display: Use Flask to render analysis results on index.html.
Input Form Handling: Accept input features through HTML forms.
Output Display: Show prediction results dynamically on the web page.

By following these steps, your Heart Health Model application will be successfully deployed and ready to predict heart health status for end users in a live environment.


## Built With and Versioning

Here are the links for the specified libraries and tools:

* [Python 3.x](https://www.python.org/) - Programming Language
* [Pandas 2.1.4](https://pandas.pydata.org/) - Data Analysis and Manipulation Library
* [Matplotlib 3.8.0](https://matplotlib.org/) - Plotting Library
* [Scikit-learn 1.2.2](https://scikit-learn.org/) - Machine Learning Library
* [SQLAlchemy 2.0.21](https://www.sqlalchemy.org/) - SQL Toolkit and Object Relational Mapper
* [Plotly 5.9.0](https://plotly.com/) - Interactive Graphing Library
* [Jupyter Notebook 1.0.0](https://jupyter.org/) - Interactive Computing Environment
* [Seaborn 0.12.2](https://seaborn.pydata.org/) - Statistical Data Visualization Library
* [Numpy 1.26.3](https://numpy.org/) - Numerical Computing Library
* [TensorFlow 2.16.1](https://www.tensorflow.org/) - Machine Learning Framework
* [Keras 3.3.3](https://keras.io/) - Deep Learning API
* [Flask 2.2.5](https://flask.palletsprojects.com/) - Web Framework for API
* [SciPy 1.11.4](https://www.scipy.org/) - Scientific Computing Library


## Authors

* **AayushiDaliparthi** - (https://github.com/AayushiDaliparthi)
* **emilyfaris** - (https://github.com/emilyfaris)
* **MaggieJane95** - (https://github.com/MaggieJane95)
* **TyliOnel** - (https://github.com/TyliOnel)

## License

This project is licensed under the © 2024 edX Boot Camps LLC License, created for the University of Toronto Continuing Studies Data Analytics Bootcamp.

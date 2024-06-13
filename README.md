# Project_4-Group_5-Heart_Health

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
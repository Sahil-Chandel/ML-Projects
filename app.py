from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.exception import CustomException

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Get the data from the form
        gender = request.form.get('gender')
        race_ethnicity = request.form.get('race_ethnicity')
        parental_level_of_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        reading_score = request.form.get('reading_score')
        writing_score = request.form.get('writing_score')

        # Check if any inputs are missing
        if None in [gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score]:
            return render_template('home.html', results="Please fill all the fields")

        # Create the custom data object
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=float(reading_score),
            writing_score=float(writing_score)
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")

        # Perform prediction
        try:
            results = predict_pipeline.predict(pred_df)
        except CustomException as e:
            print(f"Prediction error: {e}")
            return render_template('home.html', results="An error occurred during prediction.")

        print("After Prediction")
        return render_template('home.html', results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        
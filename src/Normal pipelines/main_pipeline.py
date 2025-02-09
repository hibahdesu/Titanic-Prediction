from data_ingestion import load_data
from feature_engineering import transform_df
from model_training import train_models
from model_saving import save_model
import subprocess
import os
from flask import Flask, request, jsonify, render_template
import flask

def run_pipeline():
    # Load Titanic dataset
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv?raw=True'
    df = load_data(url)

    # Apply transformations to the data
    df_transformed = transform_df(df)

    # Train the models and get the best model and evaluation reports
    best_model, reports = train_models(df_transformed)

    # Save the best model to a file
    save_model(best_model, 'best_titanic_model_with_pipeline.pkl')

    # Optional: Test output (print first few rows of the transformed data)
    print("\nTransformed data preview:")
    print(df_transformed.head())

    # Print the evaluation reports for all models
    print("\nModel evaluation reports:")
    for model_name, report in reports:
        print(f"Report for {model_name}:")
        print(report)
        print(f"###############################")

    # Now start the Flask API (this assumes that the Flask app is in a separate file, e.g., app.py)
    print("\nStarting the Flask app...")
    
    # Correct the path to app.py using os.path
    app_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'app.py')

    # Run the Flask app using the correct Python interpreter from the virtual environment
    subprocess.run([r"myenv\Scripts\python.exe", app_path])

# Add this block to ensure the function is only called when the script is run directly
if __name__ == "__main__":
    run_pipeline()

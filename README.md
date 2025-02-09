

---
![Titanic Image](https://path_to_image.com/titanic_image.jpg)

# Titanic Prediction with MLOps Pipeline

This project aims to build a machine learning model for predicting whether a passenger survived the Titanic disaster, using a structured MLOps pipeline. The pipeline covers data ingestion, feature engineering, model training, model evaluation, model saving, and serving the model through a Flask API.

### Project Structure

```plaintext
Titanic Prediction/
  ├── src/
  │     ├── Normal pipelines/
  │     │    ├── data_ingestion.py       # Data ingestion module
  │     │    ├── feature_engineering.py  # Feature engineering module
  │     │    ├── model_training.py       # Model training module
  │     │    ├── model_saving.py         # Model saving module
  │     └── api/
  │          └── app.py                  # Flask API serving the trained model
```

---

### Overview

This project uses machine learning to predict if a Titanic passenger survived based on various features such as age, fare, number of siblings, and class. The pipeline follows the MLOps steps to automate and deploy the model in a reproducible manner.

### MLOps Pipeline Steps

1. **Data Ingestion (`data_ingestion.py`)**
   - Ingest Titanic dataset from an external URL.

2. **Feature Engineering (`feature_engineering.py`)**
   - Process and transform the data, handling missing values, encoding categorical features, and scaling numerical features.

3. **Model Training (`model_training.py`)**
   - Train multiple machine learning models and evaluate their performance using metrics such as accuracy, precision, and recall.

4. **Model Saving (`model_saving.py`)**
   - Save the best-performing model along with any pre-processing pipeline into a file (`best_titanic_model_with_pipeline.pkl`).

5. **Flask API (`app.py`)**
   - Serve the trained model via a REST API endpoint where users can send a JSON payload for predictions.
   - The app also renders an HTML page (using `render_template`) for interacting with the model through a browser.

---

### Prerequisites

- Python 3.x
- Flask
- Scikit-learn
- Pandas
- NumPy
- Joblib

You can install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

Or, activate the Python virtual environment and install:

```bash
.\myenv\Scripts\activate   # For Windows
pip install -r requirements.txt
```

---

### How to Run the Pipeline

1. **Set Up the Environment:**
   Ensure that your virtual environment is activated.

   ```bash
   .\myenv\Scripts\activate  # For Windows
   ```

2. **Run the Pipeline:**
   Execute the `main_pipeline.py` script to start the end-to-end pipeline process. This will:

   - Load and preprocess the Titanic dataset.
   - Train models.
   - Save the best model.
   - Start the Flask API.

   ```bash
   python main_pipeline.py
   ```

   After this, the pipeline will run, train the model, and start the Flask app that listens for incoming requests.

---

### Flask API

Once the pipeline runs successfully, the Flask API will be started. You can access it via:

```
http://localhost:5000/
```

The API exposes the following endpoints:

1. **`/` (Home)**
   - Renders the main HTML page where users can interact with the model.

2. **`/predict` (Prediction)**
   - This endpoint accepts a POST request with JSON input and returns the survival prediction for a passenger. 
   
   Example of a POST request:

   ```json
   {
     "Age": 22,
     "Fare": 7.25,
     "Parch": 0,
     "Pclass": 3,
     "SibSp": 1
   }
   ```

   The response will contain the survival prediction:

   ```json
   {
     "prediction": "The passenger would survive the Titanic disaster."
   }
   ```

---

### File Descriptions

- **`data_ingestion.py`**: Contains the logic to load the Titanic dataset.
- **`feature_engineering.py`**: Transforms the data by handling missing values, encoding categorical features, and scaling numeric columns.
- **`model_training.py`**: Trains multiple machine learning models (e.g., Logistic Regression, Random Forest) and selects the best model.
- **`model_saving.py`**: Saves the best model and pipeline to a `.pkl` file.
- **`app.py`**: Flask application that serves the trained model, with endpoints to render HTML and make predictions.
- **`main_pipeline.py`**: The main script that runs the entire MLOps pipeline from data ingestion to serving the Flask app.

---

### How to Test the Model

To test the Flask model after running the pipeline:

1. Ensure the Flask API is running (it should automatically start when running `main_pipeline.py`).
2. Open Postman or use `curl` to send a POST request to the `/predict` endpoint with sample data.

Example using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"Age": 22, "Fare": 7.25, "Parch": 0, "Pclass": 3, "SibSp": 1}' \
    http://localhost:5000/predict
```

---

### Project Setup

1. Clone the repository or download the project folder.
2. Set up a Python virtual environment:
   ```bash
   python -m venv myenv
   ```
3. Activate the virtual environment:
   ```bash
   .\myenv\Scripts\activate  # For Windows
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the pipeline:
   ```bash
   python main_pipeline.py
   ```

---
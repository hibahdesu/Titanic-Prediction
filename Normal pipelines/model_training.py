import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Function to train and evaluate models
def train_models(df):
    # Split data into features (X) and target (y)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models with pipelines
    models = [
        (
            "LR Normal", 
            {},
            Pipeline([('scaler', StandardScaler()), ('LR', LogisticRegression())]), 
            (X_train, y_train),
            (X_test, y_test)
        ),
        (
            "LR With params", 
            {"LR__C": 1, "LR__solver": 'liblinear'},
            Pipeline([('scaler', StandardScaler()), ('LR', LogisticRegression())]), 
            (X_train, y_train),
            (X_test, y_test)
        ),
        (
            "RF Normal", 
            {},
            Pipeline([('scaler', StandardScaler()), ('RF', RandomForestClassifier())]), 
            (X_train, y_train),
            (X_test, y_test)
        ),
        (
            "RF With params", 
            {"RF__n_estimators": 30, "RF__max_depth": 3, "RF__class_weight": 'balanced'},
            Pipeline([('scaler', StandardScaler()), ('RF', RandomForestClassifier())]), 
            (X_train, y_train),
            (X_test, y_test)
        ),
        (
            "SVC With params", 
            {"SVC__C": 1, "SVC__kernel": 'linear', "SVC__class_weight": 'balanced'},
            Pipeline([('scaler', StandardScaler()), ('SVC', SVC())]), 
            (X_train, y_train),
            (X_test, y_test)
        ),
    ]

    # Initialize variables for tracking the best model
    best_accuracy = 0
    best_model = None
    best_model_name = ""

    # Set the experiment in MLflow (ensure experiment exists)
    experiment_name = "Titanics_Detection"
    try:
        # Try to get or create the experiment
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment created or already exists: {experiment_name}")
    except Exception as e:
        # If the experiment already exists, get the experiment ID
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"Experiment '{experiment_name}' already exists.")

    mlflow.set_experiment(experiment_name)

    reports = []

    # Iterate through models to train and evaluate
    for model_name, params, pipeline, (X_train, y_train), (X_test, y_test) in models:
        print(f'\nTraining and logging model: {model_name}')
        
        # Set model-specific parameters if provided
        if params:
            pipeline.set_params(**params)

        # Train the pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"Classification Report for {model_name}: \n{report}\n")

        # Log the model and metrics using MLflow
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)  # Log model parameters

            # Log accuracy metric
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            # Log the full classification report as metrics
            mlflow.log_metrics({
                'accuracy': report['accuracy'],
                'precision_class_1': report['1']['precision'] if '1' in report else 0,
                'precision_class_0': report['0']['precision'] if '0' in report else 0,
                'recall_class_1': report['1']['recall'] if '1' in report else 0,
                'recall_class_0': report['0']['recall'] if '0' in report else 0,
                'f1_score_class_1': report['1']['f1-score'] if '1' in report else 0,
                'f1_score_class_0': report['0']['f1-score'] if '0' in report else 0,
                'f1_score_macro': report['macro avg']['f1-score']
            })

            # Log the model to MLflow
            mlflow.sklearn.log_model(pipeline, "model")

            # Track the best model based on accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = pipeline
                best_model_name = model_name

        # Append the report to the list
        reports.append((model_name, report))

    # After training all models, print the best model's details
    print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Return the best model for further use
    return best_model, reports



# MLflow registration and final steps
def log_and_register_model(best_model, X_train, y_train, X_test, y_test):
    # Initialize MLflow
    mlflow.set_experiment("Titanics Detection")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Adjust if necessary

    # Start an MLflow run for logging the best model
    with mlflow.start_run(run_name="Best_Model_Run"):
        # Train the best model (assuming best_model is a pipeline)
        best_model.fit(X_train, y_train)

        # Make predictions using the best model
        y_pred = best_model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Best Model Accuracy: {accuracy:.4f}")

        # Log the accuracy as a metric
        mlflow.log_metric("accuracy", accuracy)

        # Generate the classification report and log the metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metrics({
            'accuracy': report['accuracy'],
            'precision_class_1': report['1']['precision'] if '1' in report else 0,
            'precision_class_0': report['0']['precision'] if '0' in report else 0,
            'recall_class_1': report['1']['recall'] if '1' in report else 0,
            'recall_class_0': report['0']['recall'] if '0' in report else 0,
            'f1_score_class_1': report['1']['f1-score'] if '1' in report else 0,
            'f1_score_class_0': report['0']['f1-score'] if '0' in report else 0,
            'f1_score_macro': report['macro avg']['f1-score']
        })

        # Log the model using MLflow
        mlflow.sklearn.log_model(best_model, "model")

        # After training and logging, we can register the model
        model_name = "Best_Model"  # You can customize this name
        model_uri = f'runs:/{mlflow.active_run().info.run_id}/model'
        
        # Register the model in MLflow's model registry
        mlflow.register_model(model_uri=model_uri, name=model_name)

        print(f"Model registered with name: {model_name}")

        # Optionally, you can load the model for further use
        model_version = 1  # Specify the version of the model
        model_uri = f"models:/{model_name}/{model_version}"

        # Load the registered model
        loaded_model = mlflow.sklearn.load_model(model_uri)

        # Make predictions with the loaded model
        y_pred = loaded_model.predict(X_test)

        # Display the first 4 predictions
        print("First 4 predictions from the registered model:", y_pred[:4])




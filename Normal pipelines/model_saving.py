import mlflow
import mlflow.sklearn
import pickle

# Function to save the best model as a .pkl file
def save_model(model, filename):
    # Save model locally
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

    # Also, log the model in MLflow (ensure MLflow is initialized and running)
    with mlflow.start_run(run_name="Best_Model_Save"):
        mlflow.sklearn.log_model(model, "best_model")
        print(f"Model logged in MLflow with run ID: {mlflow.active_run().info.run_id}")

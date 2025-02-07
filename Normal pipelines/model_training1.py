from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Function to train and evaluate models
def train_models(df):
    # Split data into features (X) and target (y)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVC": SVC(),
        "KNeighbors": KNeighborsClassifier(),
    }

    best_accuracy = 0
    best_model = None
    best_model_name = ""

    # Iterate through each model and evaluate performance using a pipeline
    for model_name, model in models.items():
        print(f"\nTraining {model_name} with pipeline...")

        # Create a pipeline that first scales data and then applies the model
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Feature scaling
                ("classifier", model),  # Model training
            ]
        )

        # Train the model
        pipeline.fit(X_train, y_train)

        # Evaluate the model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{model_name} Accuracy: {accuracy:.4f}")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_model_name = model_name

    print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    return best_model

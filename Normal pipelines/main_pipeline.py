from data_ingestion import load_data
from feature_engineering import transform_df
from model_training import train_models
from model_saving import save_model

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
    
# Add this block to ensure the function is only called when the script is run directly
if __name__ == "__main__":
    run_pipeline()

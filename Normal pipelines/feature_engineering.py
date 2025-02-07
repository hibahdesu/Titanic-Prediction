import pandas as pd  # Ensure pandas is imported as pd
from data_cleaning import select_number_columns, fill_missing_values_with_median

def transform_df(df: pd.DataFrame) -> pd.DataFrame:  # Use pd.DataFrame for the type hint
    print(f"Type of data before transformation: {type(df)}")
    assert isinstance(df, pd.DataFrame), f"Input data is not a DataFrame! Type: {type(df)}"
    transformed_df = fill_missing_values_with_median(select_number_columns(df))
    assert isinstance(transformed_df, pd.DataFrame), f"Output of transform_df is not a DataFrame! Type: {type(transformed_df)}"
    print(f"Type of Transformed df is: {type(transformed_df)}")
    return transformed_df
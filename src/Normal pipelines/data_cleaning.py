import math
from pandas import DataFrame


# Select number columns (features)
def select_number_columns(df: DataFrame) -> DataFrame:
    return df[["Age", "Fare", "Parch", "Pclass", "SibSp", "Survived"]]


# Fill missing values with the median
def fill_missing_values_with_median(df: DataFrame) -> DataFrame:
    for col in df.columns:
        values = sorted(df[col].dropna().tolist())
        median_value = values[math.floor(len(values) / 2)]
        df[[col]] = df[[col]].fillna(median_value)
    return df

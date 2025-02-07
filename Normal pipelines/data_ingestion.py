import pandas as pd

def load_data(url: str):
    df = pd.read_csv(url)
    return df

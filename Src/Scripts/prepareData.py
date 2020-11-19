# Basis encoding example with PennyLane
import pandas as pd
from sklearn.utils import shuffle

def normalize(DATA_PATH: str = "../../Data/Processed/data.csv"):
    """
    Normalizes the data
    """
    # Reads the data
    data = pd.read_csv(DATA_PATH)
    X, Y = data[['sex', 'cp', 'exang', 'oldpeak']].values, data['num'].values
    # normalize the data
    X = normalize(X)
    return X, Y

    
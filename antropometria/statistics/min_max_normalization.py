import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def apply_min_max_normalization(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df)  # Aplica a normalização MinMax
    df_scaled = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)  # Cria um DataFrame com os valores escalados
    return df_scaled


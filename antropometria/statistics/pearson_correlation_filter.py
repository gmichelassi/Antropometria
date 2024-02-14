import numpy as np
import pandas as pd

def apply_pearson_feature_selection(samples: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    if threshold >= 1.0 or threshold <= 0.0:
        raise ValueError(f'Expected values 0.0 < x < 1.0, received x={threshold}')
    
    # Calcula a matriz de correlação absoluta
    corr_matrix = samples.corr().abs()

    # Identifica colunas com correlação acima do limiar
    # Usa np.triu para considerar apenas o triângulo superior da matriz de correlação, evitando duplicatas
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    to_drop = [column for column in range(corr_matrix.shape[1]) if any(corr_matrix.iloc[:, column][upper_tri[:, column]] >= threshold)]
    
    # Remove as colunas identificadas
    samples_filtered = samples.drop(samples.columns[to_drop], axis=1)
    
    return samples_filtered
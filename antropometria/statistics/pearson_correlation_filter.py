import numpy as np
import pandas as pd

from scipy import stats


def apply_pearson_feature_selection(samples: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    if threshold >= 1.0 or threshold <= 0.0:
        raise ValueError(f'Expected values 0.0 < x < 1.0, received x={threshold}')

    n_features = samples.shape[1]
    features_to_delete = np.zeros(n_features, dtype=bool)

    for i in range(0, n_features):
        if not features_to_delete[i]:
            feature_i = samples.iloc[:, i].to_numpy()

            for j in range(i+1, n_features):
                if not features_to_delete[j]:
                    feature_j = samples.iloc[:, j].to_numpy()
                    pearson, pvalue = stats.pearsonr(feature_i, feature_j)
                    if abs(pearson) >= threshold:
                        features_to_delete[j] = True

    return samples[samples.columns[~features_to_delete]]

import numpy as np
import pandas as pd
from scipy import stats


class PeasonCorrelationFeatureSelector:
    def __init__(self, dataset: pd.DataFrame, threshold: float = 0.99):
        self.dataset = dataset
        self.threshold = threshold

        self.__verify_threshold()

    def apply_pearson_feature_selection(self) -> pd.DataFrame:
        n_features = self.dataset.shape[1]
        features_to_delete = np.zeros(n_features, dtype=bool)

        for i in range(0, n_features):
            if not features_to_delete[i]:
                feature_i = self.dataset.iloc[:, i].to_numpy()

                for j in range(i+1, n_features):
                    if not features_to_delete[j]:
                        feature_j = self.dataset.iloc[:, j].to_numpy()
                        pearson, _ = stats.pearsonr(feature_i, feature_j)
                        if abs(pearson) >= self.threshold:
                            features_to_delete[j] = True

        return self.dataset[self.dataset.columns[~features_to_delete]]

    def __verify_threshold(self) -> None:
        if self.threshold >= 1.0 or self.threshold <= 0.0:
            raise ValueError(f'Expected values 0.0 < x < 1.0, received x={self.threshold}')

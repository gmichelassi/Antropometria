import numpy as np
from skfeature.function.statistical_based.CFS import cfs
from sklearn.base import BaseEstimator, TransformerMixin


class CFS(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select: int = 2, mode: str = 'rank', verbose: bool = False):
        self.n_features_to_select = n_features_to_select
        self.mode = mode
        self.verbose = verbose

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ranking = cfs(x, y, self.mode)
        if self.verbose:
            print(f"Feature ranking: {ranking}")
        return ranking

    def transform(self, x: np.ndarray, ranking: np.ndarray) -> np.ndarray:
        if self.n_features_to_select is None:
            return x[:, ranking]

        return x[:, ranking[:self.n_features_to_select]]

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, **_) -> np.ndarray:
        ranking = self.fit(X, y)
        return self.transform(X, ranking)

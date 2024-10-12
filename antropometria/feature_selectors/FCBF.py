import numpy as np
from skfeature.function.information_theoretical_based.FCBF import fcbf
from sklearn.base import BaseEstimator, TransformerMixin


class FCBF(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select: int = 2, mode: str = 'rank', delta: float = 0.0, verbose: bool = False):
        self.n_features_to_select = n_features_to_select
        self.delta = delta
        self.mode = mode
        self.verbose = verbose

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ranking = fcbf(x, y, self.mode, kwargs={'delta': self.delta})
        if self.verbose:
            print(f"Feature ranking: {ranking}")
        return ranking

    def transform(self, x: np.ndarray, ranking: np.ndarray) -> np.ndarray:
        if self.n_features_to_select is None:
            return x[:, ranking]

        return x[:, ranking[:self.n_features_to_select]]

    def fit_transform(self, x: np.ndarray, y: np.ndarray = None, **fit_param) -> np.ndarray:
        ranking = self.fit(x, y)
        return self.transform(x, ranking)

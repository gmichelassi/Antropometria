import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class RFSelect(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select: int):
        self.__sel = SelectFromModel(RandomForestClassifier())
        self.n_features_to_select = n_features_to_select

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.__sel.fit(x, y)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x[:, self.__sel.get_support()]

    def fit_transform(self, x: np.ndarray, y: np.ndarray = None, **fit_param) -> np.ndarray:
        self.fit(x, y)
        return self.transform(x)

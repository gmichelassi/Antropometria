import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class RFSelect(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.__sel = SelectFromModel(RandomForestClassifier())

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.__sel.fit(x, y)

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x[:, self.__sel.get_support()]

    def fit_transform(self, x: np.ndarray, y: np.ndarray = None, **fit_param) -> np.ndarray:
        self.fit(x, y)
        return self.transform(x)

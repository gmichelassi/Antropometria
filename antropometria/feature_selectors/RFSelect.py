from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class RFSelect(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.__sel = SelectFromModel(RandomForestClassifier())

    def fit(self, x, y):
        self.__sel.fit(x, y)

        return self

    def transform(self, x):
        return x[:, self.__sel.get_support()]

    def fit_transform(self, x, y=None, **fit_param):
        self.fit(x, y)
        return self.transform(x)

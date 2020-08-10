from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.information_theoretical_based.FCBF import fcbf
from skfeature.function.statistical_based.CFS import cfs
from skfeature.function.sparse_learning_based.RFS import rfs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class mRMRProxy(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=2, mode='rank', verbose=True):
        self.n_features_to_select = n_features_to_select
        self.mode = mode
        self.verbose = verbose

    def fit(self, X, y):
        self._X = X
        self._y = y
        self.ranking_ = mrmr(self._X, self._y, self.mode)
        if self.verbose:
            print("Feature ranking: " + str(self.ranking_))
        return self

    def transform(self, X):
        return X[:, self.ranking_[:self.n_features_to_select]]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class FCBFProxy(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=2, mode='rank', delta=0.0, verbose=True):
        self.n_features_to_select = n_features_to_select
        self.delta = delta
        self.mode = mode
        self.verbose = verbose

    def fit(self, X, y):
        self._X = X
        self._y = y
        self.ranking_ = fcbf(self._X, self._y, self.mode, kwargs={'delta': self.delta})
        if self.verbose:
            print("Feature ranking: " + str(self.ranking_))
        return self

    def transform(self, X):
        return X[:, self.ranking_[:self.n_features_to_select]]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class CFSProxy(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=None, mode='rank', verbose=True):
        self.n_features_to_select = n_features_to_select
        self.mode = mode
        self.verbose = verbose

    def fit(self, X, y):
        self._X = X
        self._y = y
        self.ranking_ = cfs(self._X, self._y, self.mode)
        if self.verbose:
            print("Feature ranking: " + str(self.ranking_))
        return self

    def transform(self, X):
        if self.n_features_to_select is None:
            return X[:, self.ranking_]

        return X[:, self.ranking_[:self.n_features_to_select]]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class RFSProxy(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=None, mode='rank', gamma=1, verbose=True):
        self.n_features_to_select = n_features_to_select
        self.mode = mode
        self.verbose = verbose
        self.gamma = gamma

    def fit(self, X, y):
        self._X = X
        self._y = y
        self.ranking_ = rfs(self._X, self._y, self.mode, kwargs={'gamma': self.gamma, 'verbose': self.verbose})
        if self.verbose:
            print("Feature ranking: " + str(self.ranking_))
        return self

    def transform(self, X):
        if self.n_features_to_select is None:
            return X[:, self.ranking_]

        return X[:, self.ranking_[:self.n_features_to_select]]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class RFSelect(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.__sel = SelectFromModel(RandomForestClassifier())

    def fit(self, X, y):
        self.__sel.fit(X, y)

        return self

    def transform(self, X):
        return X[:, self.__sel.get_support()]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

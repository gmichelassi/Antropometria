from sklearn.base import BaseEstimator, TransformerMixin
from skfeature.function.information_theoretical_based.FCBF import fcbf


class FCBF(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=2, mode='rank', delta=0.0, verbose=False):
        self.n_features_to_select = n_features_to_select
        self.delta = delta
        self.mode = mode
        self.verbose = verbose

    def fit(self, x, y):
        ranking = fcbf(x, y, self.mode, kwargs={'delta': self.delta})
        if self.verbose:
            print(f"Feature ranking: {ranking}")
        return ranking

    def transform(self, x, ranking):
        if self.n_features_to_select is None:
            return x[:, ranking]

        return x[:, ranking[:self.n_features_to_select]]

    def fit_transform(self, x, y=None, **fit_param):
        ranking = self.fit(x, y)
        return self.transform(x, ranking)

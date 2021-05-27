from sklearn.base import BaseEstimator, TransformerMixin
from skfeature.function.statistical_based.CFS import cfs


class CFS(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=None, mode='rank', verbose=True):
        self.n_features_to_select = n_features_to_select
        self.mode = mode
        self.verbose = verbose

    def fit(self, x, y):
        ranking = cfs(x, y, self.mode)
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

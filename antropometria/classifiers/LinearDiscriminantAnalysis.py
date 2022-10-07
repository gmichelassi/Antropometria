from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class DiscriminantAnalysis:
    def __init__(self, name: str = 'LinearDiscriminantAnalysis', n_features: int = 0):
        self.name = name
        self.n_features = n_features
        self.estimator = LinearDiscriminantAnalysis()
        self.parameter_grid = {
            'solver': ['svd', 'lsqr', 'eigen'],
            'shrinkage': ['auto', None]
        }

    @staticmethod
    def get_trained_estimator(parameters):
        return

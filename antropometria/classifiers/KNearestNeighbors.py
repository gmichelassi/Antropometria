from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors:
    def __init__(self, name: str = 'KNeighborsClassifier', n_features: int = 0):
        self.name = name
        self.n_features = n_features
        self.estimator = KNeighborsClassifier()
        self.parameter_grid = {
            'n_neighbors': [1, 2, 3, 5, 10, 20, 25],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto'],
            'leaf_size': [1, 10, 30],
            'n_jobs': [-1]
        }

    @staticmethod
    def get_trained_estimator(parameters):
        return

from numpy import sqrt
from sklearn.ensemble import RandomForestClassifier


class RandomForests:
    def __init__(self, name: str = 'RandomForest', n_features: int = 0):
        self.name = name
        self.sqrt_n_features = int(sqrt(n_features))
        self.estimator = RandomForestClassifier()
        self.parameter_grid = {
            'n_estimators': [500, 1000, 1500, 2000],
            'criterion': ['gini'],
            'max_depth': [None],
            'min_samples_leaf': [1, 2, 5, 10, 15, 20],
            'max_features': [i for i in range(int(0.5 * self.sqrt_n_features),
                                              6 * self.sqrt_n_features,
                                              int(((6 * self.sqrt_n_features - 0.5 * self.sqrt_n_features) / 50))
                                              )],
            'bootstrap': [True],
            'n_jobs': [-1],
            'random_state': [707878],
            'class_weight': [None]
        }

    @staticmethod
    def get_trained_estimator(parameters):
        return

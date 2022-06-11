from numpy import linspace, sqrt
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self, name: str = 'RandomForest', n_features: int = 0):
        self.name = name
        self.sqrt_n_features = int(sqrt(n_features))

        self.estimator = RandomForestClassifier()
        self.parameter_grid = {
            'n_estimators': [500, 1000],
            'criterion': ['gini'],
            'max_depth': [None],
            'min_samples_leaf': [1, 2, 5, 10, 15, 20],
            'max_features': [
                self.sqrt_n_features,
                *list(
                    set(
                        [int(i) for i in
                         linspace(int(0.5 * self.sqrt_n_features), int(2 * self.sqrt_n_features), 9) if i < n_features
                         ]))
            ],
            'bootstrap': [True],
            'n_jobs': [-1],
            'random_state': [707878],
            'class_weight': [None]
        }

    @staticmethod
    def get_trained_estimator(parameters):
        return

from numpy import linspace, sqrt
from sklearn.ensemble import RandomForestClassifier
from pydash import sorted_uniq


class RandomForest:
    def __init__(self, name: str = 'RandomForest', n_features: int = 0):
        self.name = name
        self.n_features = n_features
        self.sqrt_n_features = int(sqrt(n_features))

        self.estimator = RandomForestClassifier()
        self.parameter_grid = {
            'n_estimators': [500, 1000],
            'criterion': ['gini'],
            'max_depth': [None],
            'min_samples_leaf': [1, 5, 10, 15],
            'max_features': self.max_features(),
            'bootstrap': [True],
            'n_jobs': [-1],
            'random_state': [707878],
            'class_weight': [None]
        }

    def max_features(self):
        return sorted_uniq([
            self.sqrt_n_features,
            *list(
                set(
                    [
                        int(i) for i in linspace(
                            int(0.5 * self.sqrt_n_features), int(2 * self.sqrt_n_features), 3
                        ) if i < self.n_features
                    ]
                )
            )
        ])

    @staticmethod
    def get_trained_estimator(parameters):
        return

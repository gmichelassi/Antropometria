from sklearn.svm import SVC


class SupportVectorMachine:
    def __init__(self, name: str = 'SupportVectorMachine', n_features: int = 0):
        self.name = name
        self.n_features = n_features
        self.estimator = SVC()
        self.parameter_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.01, 0.1, 1, 5, 10, 50, 100],
            'gamma': ['scale', 'auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10],
            'degree': [2, 3, 5],
            'coef0': [0],
            'probability': [True],
            'random_state': [707878]
        }

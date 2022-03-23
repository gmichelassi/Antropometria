from sklearn.neural_network import MLPClassifier


class NeuralNetwork:
    def __init__(self, name: str = 'NeuralNetwork', n_features: int = 0):
        self.name = name
        self.n_features = n_features
        self.estimator = MLPClassifier()
        self.parameter_grid = {
            'hidden_layer_sizes': [(50,),
                                   (50, 50, 50),
                                   (100,),
                                   (100, 100, 100),
                                   (n_features,),
                                   (n_features, n_features, n_features)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.001],
            'max_iter': [1000],
            'random_state': [707878]
        }

    @staticmethod
    def get_trained_estimator(parameters):
        return

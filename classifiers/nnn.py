from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import numpy as np


def make_pipes():
    pipes, models_names = [], []

    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs', 'sgd', 'adam']
    alphas = [float(j) for j in ["0." + str('0' * i) + "1" for i in range(0, 4) if i > 0]]
    learning_rates = ['constant', 'invscaling', 'adaptive']

    for activation in activations:
        for solver in solvers:
            for alpha in alphas:
                for learning_rate in learning_rates:
                    model = MLPClassifier(
                        (100, 2),
                        activation=activation,
                        solver=solver,
                        alpha=alpha,
                        learning_rate=learning_rate)
                    pipe = make_pipeline(model)
                    pipes.append(pipe)
                    models_names.append('mlpclassifier')
    return pipes, models_names


def make_grid_optimization_pipes(n_features):
    estimator = [MLPClassifier()]
    estimator_name = 'mlpclassifier'
    grid_parameters = {
        'hidden_layer_sizes': [(50, ), (50, 50, 50), (100, ), (100, 100, 100), (n_features, ), (n_features, n_features, n_features)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100],
        'learning_rate': ['adaptive'],
        'learning_rate_init': [0.001],
        'max_iter': [300],
        'random_state': [707878]
    }

    return estimator, grid_parameters, estimator_name


def make_random_optimization_pipes(n_features):
    estimator = [MLPClassifier()]
    estimator_name = 'mlpclassifier'

    random_parameters = {
        'hidden_layer_sizes': [(100, ), (n_features, ), (n_features, n_features, n_features)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd'],
        'alpha': np.random.uniform(low=0.0001, high=1, size=50),
        'learning_rate': ['adaptive'],
        'learning_rate_init': [0.001],
        'max_iter': [300]
    }

    return estimator, random_parameters, estimator_name


def make_estimator():
    estimator_name = 'mlpclassifier'
    estimator = MLPClassifier(
        (22, 22, 22),
        activation='relu',
        solver='lbfgs',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=300
    )

    return estimator, estimator_name

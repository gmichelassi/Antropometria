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


def make_grid_optimization_estimators(n_features):
    estimators = []

    hidden_layer_sizes = [(50,), (50, 50, 50), (100,), (100, 100, 100), (n_features,),
                           (n_features, n_features, n_features)]
    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs', 'sgd']
    alphas = [0.0001, 0.001, 0.01]

    for hidden_layer_size in hidden_layer_sizes:
        for activation in activations:
            for solver in solvers:
                for alpha in alphas:

                    estimator = MLPClassifier(
                        hidden_layer_sizes=hidden_layer_size,
                        activation=activation,
                        alpha=alpha,
                        solver=solver,
                        learning_rate_init=0.001,
                        max_iter=5000,
                        random_state=707878
                    )
                    parameters = [hidden_layer_size, activation, alpha, solver, 0.001, 300, 707878]
                    estimators.append((estimator, parameters))
    return estimators


def getParams():
    return ['hidden_layer_sizes', 'activation', 'alpha', 'solver', 'learning_rate_init', 'max_iter', 'random_state']


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

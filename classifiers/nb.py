from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline


def make_pipes():
    pipes, models_names = [], []

    model = GaussianNB()
    pipe = make_pipeline(model)
    pipes.append(pipe)
    models_names.append('gaussiannb')
    return pipes, models_names


def make_grid_optimization_pipes(n_features):
    estimator = GaussianNB()
    grid_parameters = {}

    return estimator, grid_parameters


def set_parameters(parameters):
    return GaussianNB()


def make_estimator():
    estimator = GaussianNB()

    return estimator

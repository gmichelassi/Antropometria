from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline


def make_pipes():
    pipes, models_names = [], []

    model = GaussianNB()
    pipe = make_pipeline(model)
    pipes.append(pipe)
    models_names.append('gaussiannb')
    return pipes, models_names


def make_grid_optimization_estimators(n_features):
    return [(GaussianNB(), [])]


def getParams():
    return []


def make_estimator():
    estimator_name = 'gaussiannb'
    estimator = GaussianNB()

    return estimator, estimator_name

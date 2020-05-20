from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from scipy.stats import randint
from scipy.stats import uniform


def make_pipes():
    pipes, models_names, models = [], [], []

    model = SVC(kernel='linear',
                C=50,
                gamma=0.1,
                degree=3,
                coef0=0.0,
                probability=True,
                random_state=1000000)
    models.append(model)

    model = SVC(kernel='poly',
                C=0.1,
                gamma=0.001,
                degree=3,
                coef0=0.0,
                probability=True,
                random_state=1000000)
    models.append(model)

    model = SVC(kernel='rbf',
                C=100,
                gamma=0.01,
                degree=3,
                coef0=0.0,
                probability=True,
                random_state=1000000)
    models.append(model)

    model = SVC(kernel='sigmoid',
                C=0.3,
                gamma=0.0004389816,
                degree=3,
                coef0=0.01,
                probability=True,
                random_state=1000000)
    models.append(model)

    for m in models:
        pipe = make_pipeline(m)
        pipes.append(pipe)
        models_names.append('svc')
    return pipes, models_names


def getParams():
    return SVC().get_params(deep=False).keys()


def make_grid_optimization_estimators(n_features):
    estimators = []

    kernels = ['linear', 'poly', 'rbf', 'sigmoid'],
    Cs = [0.01, 0.1, 1, 5, 10, 50, 100],
    gammas = ['scale', 'auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10],
    degrees = [2, 3, 5],

    for kernel in kernels:
        for C in Cs:
            for gamma in gammas:
                for degree in degrees:
                    estimator = SVC(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        degree=degree,
                        coef0=0,
                        probability=True,
                        random_state=707878
                    )
                    estimators.append(estimator)
    return estimators


def make_random_optimization_pipes(n_features):
    estimator = [SVC()]
    estimator_name = 'svc'

    random_parameters = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': uniform(loc=(2 ** -15), scale=(2 ** 15 - 2 ** -15)),
        'gamma': uniform(loc=(2 ** -15), scale=(2 ** 15 - 2 ** -15)),
        'degree': randint(2, 7),
        'coef0': [0],
        'probability': [True]
    }

    return estimator, random_parameters, estimator_name


def make_estimator():
    estimator_name = 'svc'
    estimator = SVC(
        kernel='rbf',
        C=(2 ** 12),
        gamma=(2 ** (-3)),
        degree=7,
        coef0=0,
        probability=True,
        random_state=1000000
    )

    return estimator, estimator_name

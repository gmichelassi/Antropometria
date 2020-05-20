from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from scipy.stats import randint


def make_pipes(dimensionality_reductions, n_features_to_keep):
    pipes, reductions_names, models_names = [], [], []

    random_states = [707878]
    max_features = [26]
    n_estimators = [500]
    min_sample_leafs = [1]

    dimensionality_reduction = dimensionality_reductions

    for max_feature in max_features:
        for random_state in random_states:
            for n_estimator in n_estimators:
                for min_sample_leaf in min_sample_leafs:
                    model = RandomForestClassifier(
                        n_estimators=n_estimator,
                        criterion='gini',
                        max_depth=None,
                        min_samples_leaf=min_sample_leaf,
                        max_features=max_feature,
                        bootstrap=True,
                        n_jobs=-1,
                        random_state=random_state,
                        class_weight=None)
                    pipe = make_pipeline(dimensionality_reduction, model)
                    pipes.append(pipe)
                    reductions_names.append(dimensionality_reduction.__class__.__name__)
                    models_names.append('randomforestclassifier')
    return pipes, reductions_names, models_names


def make_oob_pipes(dimensionality_reductions, n_features_to_keep):
    pipes, reductions_names, models_names = [], [], []

    random_states = [707878]
    max_features = [i for i in range(int(0.5*n_features_to_keep),
                                     6*n_features_to_keep,
                                     int((6*n_features_to_keep-0.5*n_features_to_keep)/50))]
    n_estimators = [500, 1000, 1500, 2000]
    min_sample_leafs = [1, 2, 5, 10, 15, 20]

    dimensionality_reduction = dimensionality_reductions

    for max_feature in max_features:
        for random_state in random_states:
            for n_estimator in n_estimators:
                for min_sample_leaf in min_sample_leafs:
                    model = RandomForestClassifier(
                        n_estimators=n_estimator,
                        criterion='gini',
                        max_depth=None,
                        min_samples_leaf=min_sample_leaf,
                        max_features=max_feature,
                        bootstrap=True,
                        n_jobs=-1,
                        random_state=random_state,
                        class_weight=None,
                        oob_score=True)
                    pipe = model
                    pipes.append(pipe)
                    reductions_names.append(dimensionality_reduction.__class__.__name__)
                    models_names.append('randomforestclassifier')
    return pipes, reductions_names, models_names


def make_grid_optimization_estimators(n_features):
    estimators = []

    # Parâmetros
    n_estimators = [500, 1000, 1500, 2000]
    min_samples_leaf = [1, 2, 5, 10, 15, 20]
    max_features = [i for i in range(int(0.5 * n_features), 6 * n_features, int(((6 * n_features - 0.5 * n_features) / 50)))]

    for n_estimator in n_estimators:
        for max_feature in max_features:
            for min_sample_leaf in min_samples_leaf:
                estimator = RandomForestClassifier(
                    n_estimators=n_estimator,
                    criterion='gini',
                    max_depth=None,
                    min_samples_leaf=min_sample_leaf,
                    max_features=max_feature,
                    bootstrap=True,
                    n_jobs=-1,
                    random_state=707878,
                    class_weight=None
                )
                estimators.append(estimator)
    return estimators


def getParams():
    return RandomForestClassifier().get_params(deep=False).keys()


def make_random_optimization_pipes(n_features):
    estimator = [RandomForestClassifier()]
    estimator_name = 'randomforestclassifier'

    random_parameters = {
        'n_estimators': randint(500, 2000),
        'criterion': ['gini'],
        'max_depth': [None],
        'min_samples_leaf': randint(1, 20),
        'max_features': randint(11, 77),  # desvio padrão = 10 e média igual a n_features na distribuição normal
        'bootstrap': [True],
        'n_jobs': [-1],
        'random_state': [707878],
        'class_weight': [None]
    }

    return estimator, random_parameters, estimator_name


def make_estimator():
    estimator_name = 'randomforestclassifier'
    estimator = RandomForestClassifier(
        n_estimators=500,
        criterion='gini',
        max_depth=None,
        min_samples_leaf=5,
        max_features=43,
        bootstrap=True,
        n_jobs=-1,
        random_state=707878,
        class_weight=None
    )

    return estimator, estimator_name

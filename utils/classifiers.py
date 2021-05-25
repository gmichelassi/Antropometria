from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class knn:
    def __init__(self, name):
        self.name = name

    def make_grid(self, features):
        estimator = KNeighborsClassifier()

        grid_parameters = {
            'n_neighbors': [1, 2, 3, 5, 10, 20, 25, 30, 35, 40],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'brute'],
            'leaf_size': [1, 5, 10, 20, 30],
            'n_jobs': [-1]
        }

        return estimator, grid_parameters

    def make_estimator(self, parameters):
        return KNeighborsClassifier(n_neighbors=parameters['n_neighbors'], weights=parameters['weights'],
                                    algorithm=parameters['algorithm'], leaf_size=parameters['leaf_size'], n_jobs=-1)


class nb:
    def __init__(self, name):
        self.name = name

    def make_grid(self, features):
        estimator = GaussianNB()
        grid_parameters = {}

        return estimator, grid_parameters

    def make_estimator(self, parameters):
        return GaussianNB()


class nnn:
    def __init__(self, name):
        self.name = name

    def make_grid(self, n_features):
        estimator = MLPClassifier()
        grid_parameters = {
            'hidden_layer_sizes': [(50,), (50, 50, 50), (100,), (100, 100, 100), (n_features,),
                                   (n_features, n_features, n_features)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.001],
            'max_iter': [5000],
            'random_state': [707878]
        }

        return estimator, grid_parameters

    def make_estimator(self, parameters):
        return MLPClassifier(hidden_layer_sizes=parameters['hidden_layer_sizes'], activation=parameters['activation'],
                             solver=parameters['solver'], alpha=parameters['alpha'],
                             learning_rate=parameters['learning_rate'],
                             learning_rate_init=parameters['learning_rate_init'], max_iter=5000, random_state=707878)


class rf:
    def __init__(self, name):
        self.name = name

    def make_grid(self, n_features):
        estimator = RandomForestClassifier()
        grid_parameters = {
            'n_estimators': [500, 1000, 1500, 2000],
            'criterion': ['gini'],
            'max_depth': [None],
            'min_samples_leaf': [1, 2, 5, 10, 15, 20],
            'max_features': [i for i in range(int(0.5 * n_features),
                                              6 * n_features,
                                              int(((6 * n_features - 0.5 * n_features) / 50)))],
            'bootstrap': [True],
            'n_jobs': [-1],
            'random_state': [707878],
            'class_weight': [None]
        }

        return estimator, grid_parameters

    def make_estimator(self, parameters):
        return RandomForestClassifier(n_estimators=parameters['n_estimators'], criterion=parameters['criterion'],
                                      max_depth=None,
                                      min_samples_leaf=parameters['min_samples_leaf'],
                                      max_features=parameters['max_features'],
                                      bootstrap=True, n_jobs=-1, random_state=707878, class_weight=None)


class svm:
    def __init__(self, name):
        self.name = name

    def make_grid(self, n_features):
        estimator = SVC()

        grid_parameters = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.01, 0.1, 1, 5, 10, 50, 100],
            'gamma': ['scale', 'auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10],
            'degree': [2, 3, 5],
            'coef0': [0],
            'probability': [True],
            'random_state': [707878]
        }

        return estimator, grid_parameters

    def make_estimator(self, parameters):
        return SVC(kernel=parameters['kernel'], C=parameters['C'], gamma=parameters['gamma'],
                   degree=parameters['degree'],
                   coef0=parameters['coef0'], probability=True, random_state=707878)
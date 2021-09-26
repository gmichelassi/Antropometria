import numpy as np
import pickle
import sys

from classifiers.KNearestNeighbors import KNearestNeighbors as Knn
from classifiers.NaiveBayes import NaiveBayes as Nb
from classifiers.NeuralNetwork import NeuralNetwork as Nn
from classifiers.RandomForests import RandomForests as Rf
from classifiers.SupportVectorMachine import SupportVectorMachine as Svm
from config import logger
from feature_selectors.utils.getter import get_feature_selector
from mainPreprocessing import run_preprocessing
from utils.dataset.manipulation import apply_pearson_feature_selection, apply_min_max_normalization, \
    get_difference_of_classes

log = logger.get_logger(__file__)

knn_parameters = {'n_neighbors': 1,
                  'weights': 'uniform',
                  'algorithm': 'auto',
                  'leaf_size': 1,
                  'n_jobs': -1}
nb_parameters = {}
nnn_parameters = {'hidden_layer_sizes': (50, 50, 50),
                  'activation': 'logistic',
                  'solver': 'lbfgs',
                  'alpha': 0.0001,
                  'learning_rate': 'adaptive',
                  'learning_rate_init': 0.001,
                  'max_iter': 5000,
                  'random_state': 707878}
rf_parameters = {'n_estimators': 500,
                 'criterion': 'gini',
                 'max_depth': None,
                 'min_samples_leaf': 1,
                 'max_features': 50,
                 'bootstrap': True,
                 'n_jobs': -1,
                 'random_state': 707878,
                 'class_weight': None}
svm_parameters = {'kernel': 'linear',
                  'C': 0.01,
                  'gamma': 'scale',
                  'degree': 2,
                  'coef0': 0,
                  'probability': True,
                  'random_state': 707878}


def save_model(folder, dataset_name, classes, model, p_filter, reduction, sampling, min_max):
    data = run_preprocessing(folder, dataset_name, classes, p_filter, reduction, sampling, False, min_max)

    if data is None:
        return

    x, y, synthetic_x, synthetic_y = data

    if reduction is not None:
        n = get_difference_of_classes(y)
        instances, features = x.shape
        n_features_to_keep = int(np.sqrt(features))

        if 0.0 < p_filter <= 0.99:
            x = apply_pearson_feature_selection(x, p_filter)

        if min_max:
            x = apply_min_max_normalization(x)

        x = x.values

        feature_selector = get_feature_selector(reduction, n_features_to_keep, instances, features)
        feature_selector.fit(x[:-n], y[:-n])

        log.info('Saving')
        pickle.dump(feature_selector, open('red_dim.sav', 'wb'))

    model.fit(x, y)

    log.info("Saving selected model")
    pickle.dump(model, open('output/model.sav', 'wb'))
    log.info("Done without errors!")


if __name__ == '__main__':
    args = sys.argv

    if args[1].lower() == 'knn':
        estimator = Knn().get_trained_estimator(knn_parameters)
    elif args[1].lower() == 'nb':
        estimator = Nb().get_trained_estimator(nb_parameters)
    elif args[1].lower() == 'nn':
        estimator = Nn().get_trained_estimator(nnn_parameters)
    elif args[1].lower() == 'rf':
        estimator = Rf().get_trained_estimator(rf_parameters)
    elif args[1].lower() == 'svm':
        estimator = Svm().get_trained_estimator(svm_parameters)
    else:
        estimator = None

    if estimator is not None:
        save_model('dlibHOG', 'distances_all_px_eu', ['casos', 'controles'], estimator, 0.98, 'PCA', 'Tomek', False)

import asd_data as asd
import time
import initContext as context
import random
import numpy as np
import pandas as pd
from classifiers.utils import normalizacao_min_max, build_ratio_dataset
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from config import logger

context.loadModules()
log = logger.getLogger(__file__)
random_state = 10000


def __generateRandomNumbers(how_many=1, maximum=10):
    numbers = []
    random.seed(909898)
    cont = 0
    while cont < how_many:
        nro = random.randint(0, maximum)
        if nro not in numbers:
            numbers.append(nro)
            cont += 1
    return numbers


def runRandomUnderSampling():
    X_casos, y_casos = asd.load(dataset='distances_all_px_euclidian', folder='casos',
                                feature_to_remove=['img_name', 'id'], label=1)
    X_controles, y_controles = asd.load(dataset='distances_all_px_euclidian', folder='controles',
                                        feature_to_remove=['img_name', 'id'], label=0)

    log.info("Data before random undersampling")
    log.info("Casos: {0}, {1}".format(X_casos.shape, len(y_casos)))
    log.info("Controles: {0}, {1}".format(X_controles.shape, len(y_controles)))

    number_of_features = abs(len(y_casos) - len(y_controles))
    features_to_delete = []
    if len(y_casos) > len(y_controles):
        feature_index_to_delete = __generateRandomNumbers(number_of_features, len(y_casos) - 1)

        for feature_index in feature_index_to_delete:
            features_to_delete.append(X_casos.index[feature_index])

        X_casos = X_casos.drop(features_to_delete)
        y_casos = np.delete(y_casos, feature_index_to_delete)
    else:
        feature_index_to_delete = __generateRandomNumbers(number_of_features, len(y_controles) - 1)

        for feature_index in feature_index_to_delete:
            features_to_delete.append(X_controles.index[feature_index])

        X_controles = X_controles.drop(features_to_delete)
        y_controles = np.delete(y_controles, feature_index_to_delete)

    log.info("Data after random undersampling")
    log.info("Casos: {0}, {1}".format(X_casos.shape, len(y_casos)))
    log.info("Controles: {0}, {1}".format(X_controles.shape, len(y_controles)))

    y = np.concatenate((y_casos, y_controles))
    X = asd.merge_frames([X_casos, X_controles])

    X, target = shuffle(X, y, random_state=random_state)
    return X, y


def runSmote(algorithm=''):
    X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='all', labels=False)

    log.info("Data before oversampling")
    log.info("Dataset: {0}, {1}".format(X.shape, len(y)))

    if algorithm == 'Borderline':
        log.info("Running Borderline Smote")
        X, y = BorderlineSMOTE().fit_resample(X, y)
    elif algorithm == 'KMeans':
        log.info("Running KMeans Smote")
        X, y = KMeansSMOTE().fit_resample(X, y)
    elif algorithm == 'SVM':
        log.info("Running SVM Smote")
        X, y = SVMSMOTE().fit_resample(X, y)
    else:
        log.info("Running deafult Smote")
        X, y = SMOTE().fit_resample(X, y)

    log.info("Data after oversampling")
    log.info("Dataset: {0}, {1}".format(X.shape, len(y)))

    return normalizacao_min_max(X), y


if __name__ == '__main__':
    start_time = time.time()
    X, y = asd.load(dataset='distances_all_px_euclidian', folder='controles',
                    feature_to_remove=['img_name', 'id'], label=1)

    build_ratio_dataset(X)

    # runRandomUnderSampling()
    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

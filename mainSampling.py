import asd_data as asd
import time
import initContext as context
import numpy as np
import pandas as pd
from classifiers.utils import normalizacao_min_max, build_ratio_dataset
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from config import logger
from classifiers.utils import generateRandomNumbers

context.loadModules()
log = logger.getLogger(__file__)
random_state = 10000


def runRandomUnderSampling(X, y, verbose):
    casos, controles, cont = [], [], 0
    for i in y:
        if i == 1:
            casos.append(cont)
        else:
            controles.append(cont)

        cont += 1

    X_casos, y_casos = X[casos], y[casos]
    X_controles, y_controles = X[controles], y[controles]

    if verbose:
        log.info("Data before random undersampling")
        log.info("Casos: {0}, {1}".format(X_casos.shape, len(y_casos)))
        log.info("Controles: {0}, {1}".format(X_controles.shape, len(y_controles)))

    number_of_features = abs(len(y_casos) - len(y_controles))
    features_to_delete = []
    if len(y_casos) > len(y_controles):
        feature_index_to_delete = generateRandomNumbers(number_of_features, len(y_casos) - 1)

        for feature_index in feature_index_to_delete:
            features_to_delete.append(X_casos[feature_index])

        X_casos = np.delete(X_casos, feature_index_to_delete, axis=0)
        y_casos = np.delete(y_casos, feature_index_to_delete, axis=0)
    else:
        feature_index_to_delete = generateRandomNumbers(number_of_features, len(y_controles) - 1)

        for feature_index in feature_index_to_delete:
            features_to_delete.append(X_controles[feature_index])

        X_controles = np.delete(X_controles, feature_index_to_delete, axis=0)
        y_controles = np.delete(y_controles, feature_index_to_delete, axis=0)

    if verbose:
        log.info("Data after random undersampling")
        log.info("Casos: {0}, {1}".format(X_casos.shape, len(y_casos)))
        log.info("Controles: {0}, {1}".format(X_controles.shape, len(y_controles)))

    y = np.concatenate((y_casos, y_controles))
    X = np.concatenate((X_casos, X_controles))

    X, target = shuffle(X, y, random_state=random_state)

    return X, target


def runSmote(X, y, algorithm='default', split_synthetic=False, verbose=True):
    if verbose:
        log.info("Data before oversampling")
        log.info("Dataset: {0}, {1}".format(X.shape, len(y)))

    n_casos = np.count_nonzero(y == 1)
    n_controles = np.count_nonzero(y == 0)

    N = abs(n_casos - n_controles)

    if algorithm == 'Borderline':
        if verbose:
            log.info("Running Borderline Smote")
        X_novo, y_novo = BorderlineSMOTE(random_state=random_state).fit_resample(X, y)
    elif algorithm == 'KMeans':
        if verbose:
            log.info("Running KMeans Smote")
        print(X.shape)
        print(y)
        X_novo, y_novo = KMeansSMOTE(random_state=random_state).fit_resample(X, y)
    elif algorithm == 'SVM':
        if verbose:
            log.info("Running SVM Smote")
        X_novo, y_novo = SVMSMOTE(random_state=random_state).fit_resample(X, y)
    else:
        if verbose:
            log.info("Running default Smote")
        X_novo, y_novo = SMOTE(random_state=random_state).fit_resample(X, y)

    if verbose:
        log.info("Data after oversampling")
        log.info("Dataset: {0}, {1}".format(X_novo.shape, len(y_novo)))

    if split_synthetic:
        synthetic_X = X_novo[-N:]
        synthetic_y = y_novo[-N:]

        return X, y, synthetic_X, synthetic_y
    else:
        return X_novo, y_novo, None, None


if __name__ == '__main__':
    start_time = time.time()

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

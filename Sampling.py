import asd_data as asd
import time
import initContext as context
import numpy as np
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from config import logger

context.loadModules()
log = logger.getLogger(__file__)
random_state = 10000


def runRandomUnderSampling(X, y, verbose):
    if verbose:
        log.info("Data before under sampling")
        log.info("Dataset: {0}, {1}".format(X.shape, len(y)))

    X_novo, y_novo = RandomUnderSampler(random_state=random_state).fit_resample(X, y)

    if verbose:
        log.info("Data after under sampling")
        log.info("Dataset: {0}, {1}".format(X_novo.shape, len(y_novo)))

    return X_novo, y_novo


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
        X_novo, y_novo = KMeansSMOTE(random_state=random_state, kmeans_estimator=KMeans(n_clusters=20)).fit_resample(X, y)
    elif algorithm == 'SVM':
        if verbose:
            log.info("Running SVM Smote")
        X_novo, y_novo = SVMSMOTE(random_state=random_state).fit_resample(X, y)
    elif algorithm == 'Tomek':
        if verbose:
            log.info("Running Smote Tomek")
        X_novo, y_novo = SMOTETomek(random_state=random_state).fit_resample(X, y)
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

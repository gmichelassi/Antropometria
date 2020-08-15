from utils.classifiers import rf, svm, nb, nnn, knn
import pickle
from utils.feature_selection import mRMRProxy, FCBFProxy, CFSProxy, RFSProxy, RFSelect
from sklearn.decomposition import PCA
from skrebate import ReliefF
import joblib
import numpy as np
from DimensionalityReduction import run_pre_processing, load

from config import logger
import initContext as context
context.loadModules()
log = logger.getLogger(__file__)


def __saveReduction(lib='dlibHOG', dataset='distances_all_px_eu', reduction='', filtro=0.0, min_max=False):
    X, y = load(lib, dataset, filtro, min_max, False)
    X = X.values
    instances, features = X.shape
    n_features_to_keep = int(np.sqrt(features))

    if reduction == 'PCA':
        red_dim = PCA(n_components=n_features_to_keep, whiten=True)
    elif reduction == 'mRMR':
        red_dim = mRMRProxy(n_features_to_select=n_features_to_keep, verbose=False)
    elif reduction == 'FCBF':
        red_dim = FCBFProxy(n_features_to_select=n_features_to_keep, verbose=False)
    elif reduction == 'CFS':
        red_dim = CFSProxy(n_features_to_select=n_features_to_keep, verbose=False)
    elif reduction == 'RFS':
        red_dim = RFSProxy(n_features_to_select=n_features_to_keep, verbose=False)
    elif reduction == 'ReliefF':
        red_dim = ReliefF(n_features_to_select=n_features_to_keep, n_neighbors=100, n_jobs=-1)
    elif reduction == 'RFSelect':
        red_dim = RFSelect()
    else:
        raise IOError("Dimensionality Reduction not found for parameter {0}".format(reduction))

    red_dim.fit(X, y)
    # joblib.dump(pca, 'PCA.joblib')
    pickle.dump(red_dim, open('red_dim.sav', 'wb'))


def saveModel(lib='dlibHOG', dataset='distances_all_px_eu', classifier=None, reduction=None, filtro=0.0, amostragem=None, min_max=False):
    X, y, synthetic_X, synthetic_y = run_pre_processing(lib=lib, dataset=dataset, reduction=reduction, filtro=filtro, amostragem=amostragem, split_synthetic=False, min_max=min_max)
    estimator = classifier.make_estimator()

    if reduction is not None:
        try:
            __saveReduction(lib, dataset, reduction, filtro, min_max)
        except IOError as ioe:
            log.info(ioe)

    if estimator is not None:
        log.info("Training selected model")
        estimator.fit(X, y)  # training the model

        log.info("Saving selected model")
        pickle.dump(estimator, open('model.sav', 'wb'))
        # joblib.dump(estimator, 'model.joblib')

        log.info("Done without errors!")
    else:
        log.info("It was not possible to save this model")


if __name__ == '__main__':
    saveModel(lib='dlibHOG', dataset='distances_all_px_eu', classifier=svm('svc'), reduction='PCA', filtro=0.98, amostragem='Tomek', min_max=False)

# Feature selection / Dimensionality reduction
from classifiers.custom_feature_selection import mRMRProxy, FCBFProxy, CFSProxy, RFSProxy
from sklearn.decomposition import PCA
from skrebate import ReliefF

# DataSets
import asd_data as asd

# Utils
import time
import numpy as np
import pandas as pd
import initContext as context
import mainClassBalance as sampling
from classifiers.utils import apply_pearson_feature_selection, build_ratio_dataset
from sklearn.utils.multiclass import unique_labels
from config import logger
context.loadModules()
log = logger.getLogger(__file__)


def __dimensionality_reduction(red_dim, X, y):
    reduction_name = red_dim.__class__.__name__
    log.info("Applying {0} dimensionality reduction".format(reduction_name))
    data = red_dim.fit_transform(X, y)
    return data, y


def run_dimensionality_reductions(filtro=0.0, reduction='None', amostragem='', min_max=False):
    if amostragem is not None:
        if amostragem == 'random':
            X, y, dataset = sampling.runRandomUnderSampling(min_max=min_max)
        else:
            X, y, dataset = sampling.runSmote(amostragem, min_max)
    else:
        X, y, dataset = asd.load_data(d_type='euclidian', unit='px', m='', dataset='all', labels=False)

    log.info("X.shape %s, y.shape %s", str(X.shape), str(y.shape))
    n_classes = len(unique_labels(y))

    if 0.0 < filtro <= 0.99:
        log.info("Applying pearson correlation filter")
        X = apply_pearson_feature_selection(X, filtro)
    else:
        X = X.values

    instances, features = X.shape
    log.info('Data has {0} classes, {1} instances and {2} features'.format(n_classes, instances, features))

    n_features_to_keep = int(np.sqrt(features))

    if reduction == 'None':
        log.info("Returning data without dimensionality reduction")
        return X, y
    elif reduction == 'PCA':
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
    else:
        raise IOError("Dimensionality Reduction not found for parameter {0}".format(reduction))

    return __dimensionality_reduction(red_dim, X, y)


if __name__ == '__main__':
    start_time = time.time()
    print(run_dimensionality_reductions(filtro=0.0, reduction='None'))
    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

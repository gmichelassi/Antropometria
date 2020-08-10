# Feature selection / Dimensionality reduction
from utils.feature_selection import mRMRProxy, FCBFProxy, CFSProxy, RFSProxy, RFSelect
from sklearn.decomposition import PCA
from skrebate import ReliefF

# DataSets
import asd_data as asd

# Utils
import time
import numpy as np
import pandas as pd
import initContext as context
import Sampling as sampling
from utils.utils import apply_pearson_feature_selection, build_ratio_dataset, normalizacao_min_max
from sklearn.utils.multiclass import unique_labels
from config import logger
context.loadModules()
log = logger.getLogger(__file__)


def load(lib, dataset, filtro, min_max, verbose):
    if dataset == 'distances_all_px_eu':
        if lib == 'ratio':
            X, y = asd.load_data(lib=lib, dataset=dataset, classes=['tea', 'apert', 'down', 'controles'], ratio=True, verbose=verbose)
        else:
            X, y = asd.load_data(lib=lib, dataset=dataset, classes=['tea', 'apert', 'down', 'controles'], ratio=False, verbose=verbose)
    else:
        X, y = asd.load_all(folder=lib, dataset=dataset, label_name='TEA.CTRL')

    if 0.0 < filtro <= 0.99:
        if verbose:
            log.info("Applying pearson correlation filter")
        X = apply_pearson_feature_selection(X, filtro)

    if min_max:
        if verbose:
            log.info("Applying min_max normalization")
        X = normalizacao_min_max(X)

    return X, y


def __dimensionality_reduction(red_dim, X, y, verbose):
    if red_dim is None:
        return X

    reduction_name = red_dim.__class__.__name__

    if verbose:
        log.info("Applying {0} dimensionality reduction".format(reduction_name))

    data = red_dim.fit_transform(X, y)

    return data


def run_data_processing(lib='dlibHOG', dataset='distances_all_px_eu', reduction='None', filtro=0.0, amostragem=None, split_synthetic=False, min_max=False, verbose=True):
    synthetic_X, synthetic_y = None, None

    X, y = load(lib, dataset, filtro, min_max, verbose)

    n_classes = len(unique_labels(y))

    X = X.values
    instances, features = X.shape

    if verbose:
        log.info('Data has {0} classes, {1} instances and {2} features'.format(n_classes, instances, features))

    n_features_to_keep = int(np.sqrt(features))

    if reduction == 'None':
        log.info("Not applying any dimensionality reduction")
        red_dim = None
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
    elif reduction == 'RFSelect':
        red_dim = RFSelect()
    else:
        raise IOError("Dimensionality Reduction not found for parameter {0}".format(reduction))

    X = __dimensionality_reduction(red_dim, X, y, verbose)

    if amostragem is not None:
        if amostragem == 'Random':
            X, y = sampling.runRandomUnderSampling(X, y, verbose)
        else:
            X, y, synthetic_X, synthetic_y = sampling.runSmote(X, y, amostragem, split_synthetic, verbose)

    return X, y, synthetic_X, synthetic_y


if __name__ == '__main__':
    start_time = time.time()

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

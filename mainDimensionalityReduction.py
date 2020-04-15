# Feature selection / Dimensionality reduction
from classifiers.custom_feature_selection import mRMRProxy, FCBFProxy, CFSProxy, RFSProxy
from sklearn.decomposition import PCA
from skrebate import ReliefF

# DataSets
import asd_data as asd

# Utils
from sklearn.datasets import load_wine
import time
import numpy as np
import pandas as pd
import initContext as context
from classifiers.utils import apply_pearson_feature_selection, build_ratio_dataset
from sklearn.utils.multiclass import unique_labels
from config import logger
context.loadModules()
log = logger.getLogger(__file__)


def run_dimensionality_reductions(filtro=0.0, reductionless=False):
    all_samples = {}

    # X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='all', labels=False)
    # all_samples['euclidian_px_all'] = (X, y)

    X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='all', normalization='ratio', labels=False)
    all_samples['euclidian_ratio_px_all'] = (X, y)

    # all_samples['wine'] = asd.load_wine()
    # all_samples['glass'] = asd.load_glass()

    for dataset in all_samples.keys():
        log.info("Running models for %s dataset", dataset)
        samples, labels = all_samples[dataset]

        log.info("X.shape %s, y.shape %s", str(samples.shape), str(labels.shape))
        n_classes = len(unique_labels(labels))

        if 0.0 < filtro <= 0.99:
            log.info("Applying pearson correlation filter")
            samples = apply_pearson_feature_selection(samples, filtro)
        else:
            samples = samples.values

        instances, features = samples.shape
        log.info('Data has {0} classes, {1} instances and {2} features'.format(n_classes, instances, features))

        n_features_to_keep = int(np.sqrt(features))

        dimensionality_reductions = [
            PCA(n_components=n_features_to_keep, whiten=True),
            mRMRProxy(n_features_to_select=n_features_to_keep, verbose=False),
            FCBFProxy(n_features_to_select=n_features_to_keep, verbose=False),
            CFSProxy(n_features_to_select=n_features_to_keep, verbose=False),
            RFSProxy(n_features_to_select=n_features_to_keep, verbose=False),
            ReliefF(n_features_to_select=n_features_to_keep, n_neighbors=100, n_jobs=-1)
        ]

        pd.DataFrame(data=labels).to_csv(
            path_or_buf='./data/reduction_files/labels_{0}.csv'.format(dataset),
            index=False,
            header=['labels']
        )

        log.info('Saving features without dimensionality reduction')

        if filtro != 0.0:
            path = './data/reduction_files/None_{0}_filtro {1}.csv'.format(dataset, filtro)
        else:
            path = './data/reduction_files/None_{0}.csv'.format(dataset)

        amostras = pd.DataFrame(data=samples)
        amostras.to_csv(path_or_buf=path, index=False)

        if dimensionality_reductions and not reductionless:
            log.info("Applying dimensionality reduction")
            log.info('{0} features will be kept.'.format(n_features_to_keep))

            for dimensionality_reduction in dimensionality_reductions:
                reduction_name = dimensionality_reduction.__class__.__name__
                if filtro != 0.0:
                    path = './data/reduction_files/{0}_{1}_filtro_{2}.csv'.format(reduction_name, dataset, filtro)
                else:
                    path = './data/reduction_files/{0}_{1}.csv'.format(reduction_name, dataset)

                log.info('Applying dimensionality reduction with {0}'.format(reduction_name))
                data = dimensionality_reduction.fit_transform(samples, labels)
                amostras = pd.DataFrame(data=data)
                log.info('Saving data!')
                amostras.to_csv(path_or_buf=path, index=False)
            log.info('Dimensionality reduction complete!')
        log.info('Done!')


if __name__ == '__main__':
    start_time = time.time()
    run_dimensionality_reductions(filtro=0.0, reductionless=True)
    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

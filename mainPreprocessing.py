import numpy as np

from config import logger
from feature_selectors.utils.getter import get_feature_selector
from sampling.OverSampling import OverSampling
from sampling.UnderSampling import UnderSampling
from sklearn.utils.multiclass import unique_labels
from utils.dataset.load import LoadData
from utils.dataset.manipulation import apply_pearson_feature_selection, apply_min_max_normalization

log = logger.get_logger(__file__)


def run_preprocessing(folder: str,
                      dataset_name: str,
                      classes: list,
                      p_filter: float,
                      reduction: str,
                      sampling: str,
                      apply_min_max: bool,
                      verbose: bool = True) -> (np.array, np.array):
    try:
        if verbose:
            log.info(f'Loading data from data/{folder}/{dataset_name}')
        x, y = LoadData(folder, dataset_name, classes).load()
        n_classes = len(unique_labels(y))
        instances, features = x.shape
        n_features_to_keep = int(np.sqrt(features))

        if verbose:
            log.info(f'Data has {n_classes} classes, {instances} instances and {features} features')

        if 0.0 < p_filter <= 0.99:
            if verbose:
                log.info('Applying pearson correlation filter')
            x = apply_pearson_feature_selection(x, p_filter)

        if apply_min_max:
            if verbose:
                log.info('Applying min max normalization')
            x = apply_min_max_normalization(x)

        x = x.values

        if reduction is not None:
            if verbose:
                log.info(f'Applying {reduction} reduction')
            feature_selector = get_feature_selector(reduction, n_features_to_keep, instances, features)
            feature_selector.fit_transform(x, y)

        if sampling is not None:
            if verbose:
                log.info(f'Applying {sampling} sampling')
            if sampling == 'random':
                return UnderSampling().fit_transform(x, y)
            else:
                return OverSampling(sampling).fit_transform(x, y)

        return x, y

    except IOError as ioe:
        log.error(f'Error: {ioe}')
        return None
    except ValueError as ve:
        log.error(f'Error: {ve}')
        return None

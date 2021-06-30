import numpy as np

from config import logger
from feature_selectors.utils.getter import get_feature_selector
from sampling.OverSampling import OverSampling
from sampling.UnderSampling import UnderSampling
from typing import Union
from utils.dataset.load import LoadData
from utils.dataset.manipulation import apply_pearson_feature_selection, apply_min_max_normalization

log = logger.get_logger(__file__)


def run_preprocessing(
        folder: str,
        dataset_name: str,
        classes: list,
        p_filter: float,
        reduction: str,
        sampling: str,
        apply_min_max: bool,
        verbose: bool = True
) -> Union[tuple[np.array, np.array, list], None]:

    if verbose:
        log.info(f'Loading data from data/{folder}/{dataset_name}')

    x, y = LoadData(folder, dataset_name, classes).load()
    n_classes, classes_count = np.unique(y, return_counts=True)
    instances, features = x.shape
    n_features_to_keep = int(np.sqrt(features))

    if verbose:
        log.info(f'Data has {len(n_classes)} classes, {instances} instances and {features} features')

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

        x = feature_selector.fit_transform(x, y)

    if sampling is not None:
        if len(classes_count) == 2:
            if abs(classes_count[0] - classes_count[1]) == 0:
                log.warning('Your binary dataset is balanced, please keep only `None` on SAMPLINGS constant on '
                            'mainParameterCalibration. If you don\'t, the algoritms will be executed anyway and can'
                            'slow training by a significant amount of time.')
        if verbose:
            log.info(f'Applying {sampling} sampling')
        if sampling == 'random':
            x, y = UnderSampling().fit_transform(x, y)
        else:
            x, y = OverSampling(sampling).fit_transform(x, y)

    return x, y, classes_count

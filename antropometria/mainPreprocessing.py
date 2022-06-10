import numpy as np
import os

from antropometria.config import logger
from antropometria.config.types import Reduction, Sampling
from antropometria.sampling.OverSampling import OverSampling
from antropometria.sampling.UnderSampling import UnderSampling
from antropometria.utils.dataset.load import LoadData
from antropometria.utils.dataset.manipulation import apply_pearson_feature_selection, apply_min_max_normalization
from antropometria.utils.get_feature_selector import get_feature_selector
from antropometria.utils.timeout import timeout
from typing import Tuple, List, Optional

log = logger.get_logger(__file__)


# @timeout(seconds=7500, use_timeout=(os.name.lower() != 'windows'))
def run_preprocessing(
        folder: str,
        dataset_name: str,
        classes: list,
        apply_min_max: bool = False,
        p_filter: float = 0.0,
        reduction: Optional[Reduction] = None,
        sampling: Optional[Sampling] = None,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    log.info(f'Loading data from data/{folder}/{dataset_name}') if verbose else lambda: None

    x, y = LoadData(folder, dataset_name, classes).load()
    n_classes, classes_count = np.unique(y, return_counts=True)
    instances, features = x.shape
    n_features_to_keep = int(np.sqrt(features))

    log.info(
        f'Data has {len(n_classes)} classes, {instances} instances and {features} features'
    ) if verbose else lambda: None

    if 0.0 < p_filter <= 0.99:
        log.info('Applying pearson correlation filter') if verbose else lambda: None
        x = apply_pearson_feature_selection(x, p_filter)

    if apply_min_max:
        log.info('Applying min max normalization') if verbose else lambda: None
        x = apply_min_max_normalization(x)

    x = x.values

    if reduction is not None:
        log.info(f'Applying {reduction} reduction') if verbose else lambda: None

        feature_selector = get_feature_selector(reduction, n_features_to_keep, instances, features)
        x = feature_selector.fit_transform(x, y)

    if sampling is not None:
        if len(classes_count) == 2 and abs(classes_count[0] - classes_count[1]) == 0:
            log.warning('Your binary dataset is balanced, please keep only `None` on SAMPLINGS constant on '
                        'mainParameterCalibration. If you don\'t, the algoritms will be executed anyway and can'
                        'slow parameter_calibration by a significant amount of time.')

        log.info(f'Applying {sampling} sampling') if verbose else lambda: None

        if sampling == 'Random':
            x, y = UnderSampling().fit_transform(x, y)
        else:
            x, y = OverSampling(sampling).fit_transform(x, y)

    return x, y, classes_count.tolist()

import numpy as np
import time
import platform

from antropometria.config import logger
from antropometria.config.types import Reduction, Sampling
from antropometria.preprocessing.calculate_number_of_features_to_keep import calculate_number_of_n_features_to_keep
from antropometria.sampling.OverSampling import OverSampling
from antropometria.sampling.UnderSampling import UnderSampling
from antropometria.statistics import apply_pearson_feature_selection, apply_min_max_normalization
from antropometria.feature_selectors.get_feature_selector import get_feature_selector
from antropometria.utils.timeout import timeout
from typing import Tuple, Optional

log = logger.get_logger(__file__)


@timeout(seconds=7500, use_timeout=(platform.system().lower() != 'windows'))
def preprocess(
        data,
        apply_min_max: bool = False,
        p_filter: float = 0.0,
        reduction: Optional[Reduction] = None,
        sampling: Optional[Sampling] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    log.info(f'Preprocessing data with [{reduction}, {sampling}, filter {p_filter}, min_max {apply_min_max}]')

    preprocessing_initial_time = time.time()

    x, y = data
    _, classes_count = np.unique(y, return_counts=True)
    instances, original_number_of_features = x.shape

    if 0.0 < p_filter <= 0.99:
        log.info('Applying pearson correlation filter')
        x = apply_pearson_feature_selection(x, p_filter)

    if apply_min_max:
        log.info('Applying min max normalization')
        x = apply_min_max_normalization(x)

    _, current_number_of_features = x.shape

    x = x.to_numpy()

    n_features_to_keep = calculate_number_of_n_features_to_keep(
        original_number_of_features, current_number_of_features, dataset=x
    )

    if reduction is not None:
        log.info(f'Applying {reduction} reduction')

        feature_selector = get_feature_selector(reduction, n_features_to_keep, instances, current_number_of_features)
        x = feature_selector.fit_transform(x, y)

    if sampling is not None:
        if len(classes_count) == 2 and abs(classes_count[0] - classes_count[1]) == 0:
            log.warning('Your binary dataset is balanced, please keep only `None` on SAMPLINGS constant on '
                        'mainParameterCalibration. If you don\'t, the algoritms will be executed anyway and can'
                        'slow parameter_calibration by a significant amount of time.')

        log.info(f'Applying {sampling} sampling')

        if sampling == 'Random':
            x, y = UnderSampling().fit_transform(x, y)
        else:
            x, y = OverSampling(sampling).fit_transform(x, y)

    log.info(f"Dataset shape after pre processing: {x.shape}")
    log.info(f"Preprocessing took {(time.time() - preprocessing_initial_time) / 60} minutes")

    return x, y

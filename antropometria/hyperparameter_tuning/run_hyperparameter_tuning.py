import numpy as np

from antropometria.config import get_logger
from antropometria.config import (
    BINARY,
    BINARY_FIELDNAMES,
    CLASSIFIERS,
    FILTERS,
    MIN_MAX_NORMALIZATION,
    MULTICLASS_FIELDNAMES,
    REDUCTIONS,
    SAMPLINGS,
)
from antropometria.error_estimation import run_error_estimation
from antropometria.hyperparameter_tuning.grid_search import grid_search
from antropometria.preprocessing import preprocess
from antropometria.utils.mappers import map_test_to_dict, map_grid_search_results_to_dict
from antropometria.utils.results import write_header, save_results
from antropometria.utils import skip_current_test
from itertools import product


log = get_logger(__file__)

FIELDNAMES = BINARY_FIELDNAMES if BINARY else MULTICLASS_FIELDNAMES


def run_hyperparameter_tuning(folder: str, dataset_name: str, classes: list = np.array([])):
    log.info(f'Running grid search for {folder}/{dataset_name}')

    preprocessing_params = product(REDUCTIONS, SAMPLINGS, FILTERS, MIN_MAX_NORMALIZATION)
    output_file = f'./antropometria/output/GridSearch/{folder}_{dataset_name}_best_results.csv'
    write_header(file=output_file, fieldnames=FIELDNAMES)

    for reduction, sampling, p_filter, apply_min_max in preprocessing_params:
        try:
            x, y = preprocess(data, apply_min_max, p_filter, reduction, sampling)

            for classifier in CLASSIFIERS:
                if skip_current_test(classifier.__name__, reduction):
                    log.warn(f'Skipping current test because {classifier.__name__} and {reduction} are incompatible')
                    continue

                accuracy, precision, recall, f1, parameters, best_estimator = grid_search(classifier, x, y)
                current_test = map_test_to_dict(
                    folder, classifier.__name__, reduction, p_filter, apply_min_max, sampling
                )
                grid_search_results = map_grid_search_results_to_dict(accuracy, precision, recall, precision)
                error_estimation_results = run_error_estimation(x, y, classes_count, best_estimator, sampling)

                log.info('Saving results!')
                save_results(
                    file=output_file,
                    fieldnames=FIELDNAMES,
                    dataset_shape=x.shape,
                    test=current_test,
                    grid_search_results=grid_search_results,
                    error_estimation_results=error_estimation_results,
                    parameters=parameters
                )

        except (IOError, KeyError, MemoryError, TimeoutError, ValueError) as error:
            log.error(f'Could not run current test due to error: {error}')

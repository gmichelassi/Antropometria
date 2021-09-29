import pandas as pd

import initial_context
import numpy as np
import time

from classifiers.KNearestNeighbors import KNearestNeighbors as Knn
from classifiers.NaiveBayes import NaiveBayes as Nb
from classifiers.NeuralNetwork import NeuralNetwork as Nn
from classifiers.RandomForests import RandomForests as Rf
from classifiers.SupportVectorMachine import SupportVectorMachine as Svm
from config import logger
from config.constants.general import BINARY_FIELDNAMES, MULTICLASS_FIELDNAMES, CV
from config.constants.training import \
    REDUCTIONS, SAMPLINGS, FILTERS, MIN_MAX_NORMALIZATION, SCORING, ERROR_ESTIMATION
from itertools import product
from mainPreprocessing import run_preprocessing
from sklearn.model_selection import GridSearchCV
from utils.parameter_calibration.results import write_header, get_results, save_results
from utils.parameter_calibration.special_settings import stop_running_rf, skip_current_test
from utils.parameter_calibration.mappers import map_test_to_dict, map_grid_search_results_to_dict

log = logger.get_logger(__file__)
initial_context.set_context()

CLASSIFIERS = [Svm, Nb, Knn, Rf, Nn]


def run_grid_search(
        folder: str,
        dataset_name: str,
        classes: list = np.ndarray,
        verbose: bool = True
) -> None:
    is_random_forest_done = False
    binary = True
    fieldnames = BINARY_FIELDNAMES if binary else MULTICLASS_FIELDNAMES
    output_file = f'./antropometria/output/GridSearch/{folder}_{dataset_name}_best_results.csv'

    log.info(f'Running grid search for {folder}/{dataset_name}') if verbose else lambda: None
    write_header(file=output_file, fieldnames=fieldnames)

    # tests = product(CLASSIFIERS, REDUCTIONS, SAMPLINGS, FILTERS, MIN_MAX_NORMALIZATION)
    tests = [
        # [Nn,    'ReliefF',  'Random', 0.0,  False],
        # [Nn,    'ReliefF',  'Random', 0.0,  True],
        # [Nn,    'CFS',      'Random', 0.0,  False],
        # [Nn,    'RFSelect', 'Random', 0.0,  False],
        # [Nn,    'RFSelect', 'Random', 0.0,  True],
        # [Nn,    'PCA',      'Random', 0.0,  False],
        # [Nn,    'PCA',      'Random', 0.0,  True],
        # [Knn,   'ReliefF',  'Random', 0.0,  False],
        # [Knn,   'ReliefF',  'Random', 0.0,  True],
        # [Nn,    'ReliefF',  None,     0.0,  False],
        # [Nn,    'ReliefF',  None,     0.0,  True],
        # [Rf,    None,       'Random', 0.0,  False],
        [Nn,    'CFS',      'Random', 0.0,  True],
        [Nn,    'FCBF',     'Random', 0.0,  True],
        [Nn,    'mRMR',     'Random', 0.0,  False],
        [Rf,    None,       'Random', 0.0,  True],
        [Nn,    'mRMR',     'Random', 0.0,  True],
    ]
    for classifier, reduction, sampling, p_filter, min_max in tests:
        try:
            current_test_initial_time = time.time()
            if skip_current_test(is_random_forest_done, classifier.__name__, reduction):
                continue

            log.info(
                f'Running test with'
                f' classifier: {classifier.__name__},'
                f' reduction: {reduction},'
                f' sampling: {sampling},'
                f' filtro: {p_filter},'
                f' min_max: {min_max}'
            ) if verbose else lambda: None

            preprocessing_initial_time = time.time()
            x, y, classes_count = run_preprocessing(
                folder=folder,
                dataset_name=dataset_name,
                classes=classes,
                apply_min_max=min_max,
                p_filter=p_filter,
                reduction=reduction,
                sampling=sampling
            )
            log.debug(f"Pre processing took {(time.time() - preprocessing_initial_time) / 60} minutes")

            instances, features = x.shape
            model = classifier(n_features=features)

            log.info('Running cross validation') if verbose else lambda: None

            grd = GridSearchCV(
                estimator=model.estimator,
                param_grid=model.parameter_grid,
                scoring=SCORING,
                cv=CV,
                refit='f1',
                n_jobs=-1
            )

            grid_results = grd.fit(x, y)

            accuracy, precision, recall, f1, parameters = get_results(grid_results)
            current_test = map_test_to_dict(folder, model.name, reduction, p_filter, min_max, sampling)
            grid_search_results = map_grid_search_results_to_dict(accuracy, precision, recall, precision)

            log.info(
                f'Best result for test [{model.name}, {reduction}, {sampling}, {p_filter}, {min_max}] '
                f'with f1-score {(f1 * 100):.2f}%.'
            ) if verbose else lambda: None
            log.info(f'Best parameters found: {parameters}') if verbose else lambda: None

            log.info(f'Running error estimation')
            error_estimation = ERROR_ESTIMATION[str(sampling)](x, y, classes_count, grid_results.best_estimator_)
            error_estimation_results = error_estimation.run_error_estimation()

            log.info('Saving results!') if verbose else lambda: None
            save_results(
                file=output_file,
                fieldnames=fieldnames,
                test=current_test,
                grid_search_results=grid_search_results,
                error_estimation_results=error_estimation_results,
                parameters=parameters
            )

            current_test_ellapsed_time = (time.time() - current_test_initial_time) / 60

            log.info(f"Finished current test in {current_test_ellapsed_time:.2f} minutes") if verbose else lambda: None
        except (IOError, KeyError, MemoryError, TimeoutError, ValueError) as error:
            log.error(f'Could not run current test due to error: {error}')
        finally:
            is_random_forest_done = stop_running_rf(
                is_random_forest_done,
                classifier.__name__,
                reduction
            )


def main():
    start_time = time.time()
    run_grid_search('dlibHOG', 'dlibhog95', [''])
    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))


if __name__ == '__main__':
    main()

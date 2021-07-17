import csv
import initial_context
import numpy as np
import time

from config.constants import CV, EMPTY_ERROR_ESTIMATION_DICT, FIELDNAMES, FILTERS, MIN_MAX_NORMALIZATION, REDUCTIONS, \
    SAMPLINGS, SCORING
from classifiers.KNearestNeighbors import KNearestNeighbors as Knn
from classifiers.NaiveBayes import NaiveBayes as Nb
from classifiers.NeuralNetwork import NeuralNetwork as Nn
from classifiers.RandomForests import RandomForests as Rf
from classifiers.SupportVectorMachine import SupportVectorMachine as Svm
from config import logger
from mainPreprocessing import run_preprocessing
from sklearn.model_selection import GridSearchCV
from utils.training.retraining import error_estimation
from utils.training.special_settings import stop_running_rf, skip_current_test

log = logger.get_logger(__file__)
initial_context.set_context()

CLASSIFIERS = [Knn, Nb, Nn, Rf, Svm]


def run_grid_search(folder: str, dataset_name: str, classes: list = np.array([])):
    log.info(f'Running grid search for {folder}/{dataset_name}')

    is_random_forest_done = False

    with open(f'./antropometria/output/GridSearch/{folder}_{dataset_name}_best_results.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

    for classifier in CLASSIFIERS:
        for reduction in REDUCTIONS:
            for sampling in SAMPLINGS:
                for p_filter in FILTERS:
                    for min_max in MIN_MAX_NORMALIZATION:
                        current_test_initial_time = time.time()
                        if skip_current_test(is_random_forest_done, classifier, reduction):
                            continue

                        log.info(
                            f'Running test with '
                            f'classifier: {classifier.__name__}, '
                            f'reduction: {reduction}, '
                            f'sampling: {sampling}, '
                            f'filtro: {p_filter}, '
                            f'min_max: {min_max}'
                        )

                        try:
                            data = run_preprocessing(
                                folder,
                                dataset_name,
                                classes,
                                p_filter,
                                reduction,
                                sampling,
                                min_max
                            )
                        except (IOError, ValueError, MemoryError) as error:
                            log.error(f'Could not run current test due to error: {error}')
                            continue

                        x, y, classes_count = data
                        instances, features = x.shape

                        model = classifier(n_features=features)

                        log.info('Running cross validation')
                        try:
                            grd = GridSearchCV(
                                estimator=model.estimator,
                                param_grid=model.parameter_grid,
                                scoring=SCORING,
                                cv=CV,
                                refit='f1',
                                n_jobs=-1
                            )

                            grid_results = grd.fit(x, y)
                        except (KeyError, ValueError) as error:
                            log.error(f'Could not run cross validation: {error}')
                            grid_results = None

                        if grid_results is not None:
                            f1 = grid_results.best_score_
                            precision = grid_results.cv_results_['mean_test_precision'][grid_results.best_index_]
                            recall = grid_results.cv_results_['mean_test_recall'][grid_results.best_index_]
                            accuracy = grid_results.cv_results_['mean_test_accuracy'][grid_results.best_index_]
                            parameters = grid_results.best_params_

                            log.info(f'Best result for test [{model.name}, {reduction}, {sampling}, {p_filter}, '
                                     f'{min_max}] with f1-score {(f1 * 100):.2f}%.')
                            log.info(f'Best parameters found: {parameters}')

                            if sampling is not None and sampling != 'Random':
                                log.info(f'Running error estimation')
                                error_estimation_results = error_estimation(x, y, classes_count,
                                                                            grid_results.best_estimator_)
                            else:
                                error_estimation_results = EMPTY_ERROR_ESTIMATION_DICT

                            log.info('Saving results!')
                            with open(f'./antropometria/output/GridSearch/{folder}_{dataset_name}_best_results.csv',
                                      'a') as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
                                row = {
                                    'biblioteca': folder,
                                    'classifier': model.name,
                                    'reduction': reduction,
                                    'filtro': p_filter,
                                    'min_max': min_max,
                                    'balanceamento': sampling,
                                    'cv_accuracy': accuracy,
                                    'cv_precision': precision,
                                    'cv_recall': recall,
                                    'cv_f1score': f1,
                                    'parameters': parameters
                                }
                                row.update(error_estimation_results)
                                writer.writerow(row)

                        is_random_forest_done = stop_running_rf(is_random_forest_done, model.name, reduction)
                        current_test_ellapsed_time = (time.time() - current_test_initial_time) / 60
                        log.info(f"Finished current test in {current_test_ellapsed_time:.2f} minutes")


if __name__ == '__main__':
    start_time = time.time()
    run_grid_search('dlibHOG', 'distances_all_px_eu', ['casos', 'controles'])
    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

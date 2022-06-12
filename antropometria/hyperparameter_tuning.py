import numpy as np
import time

from antropometria import initial_context
from antropometria.classifiers import (
    KNearestNeighbors as Knn,
    NaiveBayes as Nb,
    NeuralNetwork as Nn,
    RandomForest as Rf,
    SupportVectorMachine as Svm
)
from antropometria.config import get_logger
from antropometria.config import (
    BINARY,
    BINARY_FIELDNAMES,
    CV,
    FILTERS,
    MIN_MAX_NORMALIZATION,
    MULTICLASS_FIELDNAMES,
    REDUCTIONS,
    SAMPLINGS,
    SCORING
)
from antropometria.error_estimation import run_error_estimation
from antropometria.preprocessing import run_preprocessing
from antropometria.utils.mappers import map_test_to_dict, map_grid_search_results_to_dict
from antropometria.utils.results import write_header, get_results, save_results
from antropometria.utils import skip_current_test
from itertools import product
from sklearn.model_selection import GridSearchCV

log = get_logger(__file__)
initial_context.set_context()

CLASSIFIERS = [Svm, Nn, Rf, Knn, Nb]
FIELDNAMES = BINARY_FIELDNAMES if BINARY else MULTICLASS_FIELDNAMES


def grid_search(classifier, x, y, classes_count, verbose: bool = True):
    log.info(f'Running cross validation for {classifier.__name__}') if verbose else lambda: None

    initial_time = time.time()
    n_instances, n_features = x.shape
    model = classifier(n_features=n_features)

    grd = GridSearchCV(
        estimator=model.estimator,
        param_grid=model.parameter_grid,
        scoring=SCORING,
        cv=CV,
        refit='f1',
        n_jobs=-1
    )
    grid_results = grd.fit(x, y)

    accuracy, precision, recall, f1, parameters, best_estimator = get_results(grid_results)
    ellapsed_time = (time.time() - initial_time) / 60

    log.info(f"Finished current grid search in {ellapsed_time:.2f} minutes") if verbose else lambda: None
    log.info(f'Results presented f1-score {(f1 * 100):.2f}%.') if verbose else lambda: None
    log.info(f'Best parameters found: {parameters}') if verbose else lambda: None

    return accuracy, precision, recall, f1, parameters, best_estimator


def hyperparameter_tuning(
        folder: str,
        dataset_name: str,
        classes: list = np.array([]),
        verbose: bool = True
) -> None:
    log.info(f'Running grid search for {folder}/{dataset_name}') if verbose else lambda: None

    preprocessing_params = product(REDUCTIONS, SAMPLINGS, FILTERS, MIN_MAX_NORMALIZATION)
    output_file = f'./antropometria/output/GridSearch/{folder}_{dataset_name}_best_results.csv'
    write_header(file=output_file, fieldnames=FIELDNAMES)

    for reduction, sampling, p_filter, apply_min_max in preprocessing_params:
        try:
            x, y, classes_count = run_preprocessing(
                folder, dataset_name, classes, apply_min_max, p_filter, reduction, sampling, verbose
            )

            for classifier in CLASSIFIERS:
                if skip_current_test(classifier.__name__, reduction):
                    continue

                accuracy, precision, recall, f1, parameters, best_estimator = grid_search(classifier, x, y, classes_count)
                current_test = map_test_to_dict(folder, classifier.__name__, reduction, p_filter, apply_min_max, sampling)
                grid_search_results = map_grid_search_results_to_dict(accuracy, precision, recall, precision)
                error_estimation_results = run_error_estimation(x, y, classes_count, best_estimator, sampling, verbose)

                log.info('Saving results!') if verbose else lambda: None
                save_results(
                    file=output_file,
                    fieldnames=FIELDNAMES,
                    test=current_test,
                    grid_search_results=grid_search_results,
                    error_estimation_results=error_estimation_results,
                    parameters=parameters
                )

        except (IOError, KeyError, MemoryError, TimeoutError, ValueError) as error:
            log.error(f'Could not run current test due to error: {error}')


def main():
    start_time = time.time()

    hyperparameter_tuning('openface', 'distances_px', ['casos', 'controles'])

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

import csv

from typing import Tuple


def write_header(file: str, fieldnames: list[str]) -> None:
    with open(file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def get_results(grid_results) -> Tuple[float, float, float, float, dict]:
    f1 = grid_results.best_score_
    precision = grid_results.cv_results_['mean_test_precision'][grid_results.best_index_]
    recall = grid_results.cv_results_['mean_test_recall'][grid_results.best_index_]
    accuracy = grid_results.cv_results_['mean_test_accuracy'][grid_results.best_index_]
    parameters = grid_results.best_params_

    return accuracy, precision, recall, f1, parameters


def save_results(
        file: str,
        fieldnames: list[str],
        test: dict,
        grid_search_results: dict,
        error_estimation_results: dict,
        parameters: dict
) -> None:
    with open(file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        results = {}
        results.update(test)
        results.update(grid_search_results)
        results.update(error_estimation_results)
        results['parameters'] = parameters
        writer.writerow(results)


import numpy as np
import os
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from antropometria.utils.training.ErrorEstimation import ErrorEstimation
from antropometria.config.constants import CV
from typing import Any, Tuple


class DefaultErrorEstimation(ErrorEstimation):
    def __init__(self, x: np.ndarray, y: np.ndarray, class_count: list[int], estimator: Any, sampling: str):
        super(DefaultErrorEstimation, self).__init__(x, y, class_count, estimator, sampling)

    def run_error_estimation(self) -> dict[str, tuple[float, float]]:
        folds = self.get_folds()
        accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro \
            = self.calculate_metrics(folds)

        return self.calculate_results(
            accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro
        )

    def get_folds(self) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        folds, current_fold = [], 0

        for train_index, test_index in CV.split(self.x, self.y):
            x_train: np.ndarray = self.x[train_index]
            y_train: np.ndarray = self.y[train_index]
            x_test: np.ndarray = self.x[test_index]
            y_test: np.ndarray = self.y[test_index]

            folds.append((x_train, y_train, x_test, y_test))
            current_fold += 1

        return folds

import numpy as np
import os
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from antropometria.utils.training.ErrorEstimation import ErrorEstimation
from antropometria.config.constants import CV, N_SPLITS
from antropometria.utils.dataset.manipulation import get_difference_of_classes

from typing import Any, Tuple


class SmoteErrorEstimation(ErrorEstimation):
    def __init__(self, x: np.ndarray, y: np.ndarray, class_count: list[int], estimator: Any, sampling: str):
        super(SmoteErrorEstimation, self).__init__(x, y, class_count, estimator, sampling)

        self.diff_classes = get_difference_of_classes(self.class_count)
        self.x: np.ndarray = x[:-self.diff_classes]
        self.y: np.ndarray = y[:-self.diff_classes]
        self.synthetic_x: np.ndarray = x[-self.diff_classes:] if self.diff_classes > 0 else np.array([])
        self.synthetic_y: np.ndarray = y[-self.diff_classes:] if self.diff_classes > 0 else np.array([])
        self.splited_synthetic_x = np.array_split(self.synthetic_x, N_SPLITS)
        self.splited_synthetic_y = np.array_split(self.synthetic_y, N_SPLITS)

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

            x_train, y_train = self.complete_fold(x_train, y_train, current_fold)
            folds.append((x_train, y_train, x_test, y_test))
            current_fold += 1

        return folds

    def complete_fold(self, x: np.ndarray, y: np.ndarray, current_fold: int) -> Tuple[np.ndarray, np.ndarray]:
        completed_x: np.ndarray = x
        completed_y: np.ndarray = y
        for i in range(N_SPLITS):
            if i != current_fold:
                for j in range(len(self.splited_synthetic_x[i])):
                    completed_x = np.append(arr=completed_x, values=[self.splited_synthetic_x[i][j]], axis=0)
                    completed_y = np.append(arr=completed_y, values=[self.splited_synthetic_y[i][j]], axis=0)

        return completed_x, completed_y

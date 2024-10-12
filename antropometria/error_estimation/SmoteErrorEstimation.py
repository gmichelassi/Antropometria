from typing import Any, Dict, List, Tuple

import numpy as np

from antropometria.config.constants import CV, N_SPLITS
from antropometria.error_estimation.ErrorEstimation import ErrorEstimation
from antropometria.utils.get_difference_of_classes import \
    get_difference_of_classes


class SmoteErrorEstimation(ErrorEstimation):
    def __init__(self, x: np.ndarray, y: np.ndarray, class_count: List[int], estimator: Any):
        super(SmoteErrorEstimation, self).__init__(x, y, class_count, estimator)

        self.diff_classes = get_difference_of_classes(self.class_count)
        self.x: np.ndarray = x[:-self.diff_classes]
        self.y: np.ndarray = y[:-self.diff_classes]
        self.synthetic_x: np.ndarray = x[-self.diff_classes:] if self.diff_classes > 0 else np.array([])
        self.synthetic_y: np.ndarray = y[-self.diff_classes:] if self.diff_classes > 0 else np.array([])
        self.splitted_synthetic_x = np.array_split(self.synthetic_x, N_SPLITS)
        self.splitted_synthetic_y = np.array_split(self.synthetic_y, N_SPLITS)

    def run_error_estimation(self) -> Dict[str, Tuple[float, float]]:
        folds = self.get_folds()

        if self.binary:
            accuracy, precision, recall, f1 = self.calculate_binary_metrics(folds)

            return self.calculate_binary_results(accuracy, precision, recall, f1)

        accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro \
            = self.calculate_multiclass_metrics(folds)

        return self.calculate_multiclass_results(
            accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro
        )

    def get_folds(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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
                for j in range(len(self.splitted_synthetic_x[i])):
                    completed_x = np.append(arr=completed_x, values=[self.splitted_synthetic_x[i][j]], axis=0)
                    completed_y = np.append(arr=completed_y, values=[self.splitted_synthetic_y[i][j]], axis=0)

        return completed_x, completed_y

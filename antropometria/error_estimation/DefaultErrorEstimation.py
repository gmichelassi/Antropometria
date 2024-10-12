from typing import Any, Dict, List, Tuple

import numpy as np

from antropometria.config.constants import CV
from antropometria.error_estimation.ErrorEstimation import ErrorEstimation


class DefaultErrorEstimation(ErrorEstimation):
    def __init__(self, x: np.ndarray, y: np.ndarray, class_count: List[int], estimator: Any):
        super(DefaultErrorEstimation, self).__init__(x, y, class_count, estimator)

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
        folds = []

        for train_index, test_index in CV.split(self.x, self.y):
            x_train: np.ndarray = self.x[train_index]
            y_train: np.ndarray = self.y[train_index]
            x_test: np.ndarray = self.x[test_index]
            y_test: np.ndarray = self.y[test_index]

            folds.append((x_train, y_train, x_test, y_test))

        return folds

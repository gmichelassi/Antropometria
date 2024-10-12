from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from antropometria.config.constants import (CV, N_SPLITS,
                                            TEMPORARY_RANDOM_SAMPLES,
                                            TEMPORARY_RANDOM_SAMPLES_LABELS)
from antropometria.error_estimation.ErrorEstimation import ErrorEstimation


class RandomSamplingErrorEstimation(ErrorEstimation):
    def __init__(self, x: np.ndarray, y: np.ndarray, class_count: List[int], estimator: Any):
        super(RandomSamplingErrorEstimation, self).__init__(x, y, class_count, estimator)
        self.removed_values = pd.read_csv(TEMPORARY_RANDOM_SAMPLES).to_numpy()
        self.removed_values_labels = pd.read_csv(TEMPORARY_RANDOM_SAMPLES_LABELS).T.to_numpy()[0]

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
        splitted_values = np.array_split(self.removed_values, N_SPLITS, axis=0)
        splitted_values_labels = np.array_split(self.removed_values_labels, N_SPLITS, axis=0)

        for train_index, test_index in CV.split(self.x, self.y):
            x_train: np.ndarray = self.x[train_index]
            y_train: np.ndarray = self.y[train_index]
            x_test: np.ndarray = self.x[test_index]
            y_test: np.ndarray = self.y[test_index]

            x_test = np.append(x_test, splitted_values[current_fold], axis=0)
            y_test = np.append(y_test, splitted_values_labels[current_fold], axis=0)

            folds.append((x_train, y_train, x_test, y_test))
            current_fold += 1

        return folds

import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from antropometria.config.constants.general import \
    TEMPORARY_RANDOM_SAMPLES, TEMPORARY_RANDOM_SAMPLES_LABELS, N_SPLITS, CV
from antropometria.utils.error_estimation.ErrorEstimation import ErrorEstimation
from typing import Any, Tuple


class RandomSamplingErrorEstimation(ErrorEstimation):
    def __init__(self, x: np.ndarray, y: np.ndarray, class_count: list[int], estimator: Any, sampling: str):
        super(RandomSamplingErrorEstimation, self).__init__(x, y, class_count, estimator, sampling)
        self.removed_values = pd.read_csv(TEMPORARY_RANDOM_SAMPLES).to_numpy()
        self.removed_values_labels = pd.read_csv(TEMPORARY_RANDOM_SAMPLES_LABELS).T.to_numpy()[0]

    def run_error_estimation(self) -> dict[str, tuple[float, float]]:
        folds = self.get_folds()
        accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro \
            = self.calculate_metrics(folds)

        return self.calculate_results(
            accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro
        )

    def get_folds(self) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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

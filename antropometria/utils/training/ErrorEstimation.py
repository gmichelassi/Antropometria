import math
import numpy as np
import os
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from antropometria.config.constants import CV, EMPTY_ERROR_ESTIMATION_DICT, N_SPLITS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from antropometria.utils.metrics import calculate_mean, calculate_std
from antropometria.utils.dataset.manipulation import get_difference_of_classes
from typing import Any, Tuple


class ErrorEstimation:
    def __init__(self, x: np.ndarray, y: np.ndarray, class_count: list[int], estimator: Any):
        self.class_count = class_count
        self.estimator = estimator
        self.diff_classes = get_difference_of_classes(self.class_count)

        self.x: np.ndarray = x[:-self.diff_classes]
        self.y: np.ndarray = y[:-self.diff_classes]
        self.synthetic_x: np.ndarray = x[-self.diff_classes:] if self.diff_classes > 0 else np.array([])
        self.synthetic_y: np.ndarray = y[-self.diff_classes:] if self.diff_classes > 0 else np.array([])
        self.splited_synthetic_x = np.array_split(self.synthetic_x, N_SPLITS)
        self.splited_synthetic_y = np.array_split(self.synthetic_y, N_SPLITS)

    def run_error_estimation(self):
        pass

    def split_dataset(self) -> list[np.ndarray]:
        folds, current_fold = [], 0

        for train_index, test_index in CV.split(self.x, self.y):
            x_train, y_train = self.x[train_index], self.y[train_index]
            x_test, y_test = self.x[test_index], self.y[test_index]

            x_train, y_train = self.complete_fold(current_fold)
            folds.append((x_train, y_train, x_test, y_test))
            current_fold += 1

        return folds

    def complete_fold(self, x: np.ndarray, y: np.ndarray, current_fold: int) -> Tuple[np.ndarray, np.ndarray]:
        completed_x = self.x
        completed_y = self.y
        for i in range(N_SPLITS):
            if i != current_fold:
                for j in range(len(splited_synthetic_x[i])):
                    completed_x = np.append(arr=self.x, values=[splited_synthetic_x[i][j]], axis=0)
                    completed_y = np.append(arr=self.y, values=[splited_synthetic_y[i][j]], axis=0)

        return completed_x, completed_y

    @staticmethod
    def calculate_confidence_interval(metric: list, mean_metric: float) -> Tuple[float, float]:
        tc = 2.262
        std = calculate_std(np.array(metric))

        ic_lower = mean_metric - tc * (std / math.sqrt(N_SPLITS))
        ic_upper = mean_metric + tc * (std / math.sqrt(N_SPLITS))

        return ic_lower, ic_upper


def complete_fold(
        x: np.ndarray,
        y: np.ndarray,
        synthetic_x: np.ndarray,
        synthetic_y: np.ndarray,
        current_fold: int
) -> Tuple[np.ndarray, np.ndarray]:
    synthetic_x = np.array_split(synthetic_x, N_SPLITS)
    synthetic_y = np.array_split(synthetic_y, N_SPLITS)

    for i in range(N_SPLITS):
        if i != current_fold:
            for j in range(len(synthetic_x[i])):
                x = np.append(arr=x, values=[synthetic_x[i][j]], axis=0)
                y = np.append(arr=y, values=[synthetic_y[i][j]], axis=0)

    return x, y


def error_estimation(
        x: np.ndarray,
        y: np.ndarray,
        classes_count: list,
        estimator: Any
) -> dict:
    n = get_difference_of_classes(classes_count)

    if n == 0:
        return EMPTY_ERROR_ESTIMATION_DICT

    synthetic_x = x[-n:]
    synthetic_y = y[-n:]
    x = x[:-n]
    y = y[:-n]

    current_fold = 0
    folds = []
    for train_index, test_index in CV.split(x, y):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        x_train, y_train = complete_fold(x_train, y_train, synthetic_x, synthetic_y, current_fold)
        folds.append((x_train, y_train, x_test, y_test))
        current_fold += 1

    accuracy = []
    precision_micro = []
    recall_micro = []
    f1_micro = []
    precision_macro = []
    recall_macro = []
    f1_macro = []

    for i in range(N_SPLITS):
        estimator.fit(folds[i][0], folds[i][1])
        y_predict = estimator.predict(folds[i][2])

        accuracy.append(accuracy_score(y_true=folds[i][3], y_pred=y_predict))

        precision_micro.append(precision_score(y_true=folds[i][3], y_pred=y_predict, average='micro'))
        recall_micro.append(recall_score(y_true=folds[i][3], y_pred=y_predict, average='micro'))
        f1_micro.append(f1_score(y_true=folds[i][3], y_pred=y_predict, average='micro'))

        precision_macro.append(precision_score(y_true=folds[i][3], y_pred=y_predict, average='macro'))
        recall_macro.append(recall_score(y_true=folds[i][3], y_pred=y_predict, average='macro'))
        f1_macro.append(f1_score(y_true=folds[i][3], y_pred=y_predict, average='macro'))

    mean_results = calculate_mean({
        'accuracy': accuracy,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    })

    f1_micro_ic = calculate_confidence_interval(f1_micro, mean_results['f1_micro'])
    f1_macro_ic = calculate_confidence_interval(f1_macro, mean_results['f1_macro'])

    return {
        'err_accuracy': mean_results['accuracy'],
        'err_precision_micro': mean_results['precision_micro'],
        'err_recall_micro': mean_results['recall_micro'],
        'err_f1score_micro': mean_results['f1_micro'],
        'err_f1micro_ic': f1_micro_ic,
        'err_precision_macro': mean_results['precision_macro'],
        'err_recall_macro': mean_results['recall_macro'],
        'err_f1score_macro': mean_results['f1_macro'],
        'err_f1macro_ic': f1_macro_ic,
    }

import math
import numpy as np
import os
import sys

from antropometria.config.constants.general import N_SPLITS

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from abc import ABC, abstractmethod
from antropometria.utils.metrics import calculate_mean, calculate_std
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Any, List, Tuple


class ErrorEstimation(ABC):
    def __init__(self, x: np.ndarray, y: np.ndarray, class_count: List[int], estimator: Any):
        self.class_count = class_count
        self.estimator = estimator
        self.x = x
        self.y = y

    @abstractmethod
    def run_error_estimation(self) -> dict[str, Tuple[float, float]]:
        pass

    @abstractmethod
    def get_folds(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        pass

    def calculate_metrics(self, folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) \
            -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
        accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro = \
            [], [], [], [], [], [], []

        for i in range(N_SPLITS):
            x_train, y_train = folds[i][0], folds[i][1]
            x_test, y_test = folds[i][2], folds[i][3]

            self.estimator.fit(x_train, y_train)
            y_predict = self.estimator.predict(x_test)

            accuracy.append(accuracy_score(y_true=y_test, y_pred=y_predict))
            precision_micro.append(precision_score(y_true=y_test, y_pred=y_predict, average='micro'))
            recall_micro.append(recall_score(y_true=y_test, y_pred=y_predict, average='micro'))
            f1_micro.append(f1_score(y_true=y_test, y_pred=y_predict, average='micro'))
            precision_macro.append(precision_score(y_true=y_test, y_pred=y_predict, average='macro'))
            recall_macro.append(recall_score(y_true=y_test, y_pred=y_predict, average='macro'))
            f1_macro.append(f1_score(y_true=y_test, y_pred=y_predict, average='macro'))

        return accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro

    def calculate_results(
            self,
            accuracy: List[float],
            precision_micro: List[float],
            recall_micro: List[float],
            f1_micro: List[float],
            precision_macro: List[float],
            recall_macro: List[float],
            f1_macro: List[float]
    ) -> dict[str, Tuple[float, float]]:
        mean_results = calculate_mean({
            'accuracy': accuracy,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro
        })
        f1_micro_ic = self.calculate_confidence_interval(f1_micro, mean_results['f1_micro'])
        f1_macro_ic = self.calculate_confidence_interval(f1_macro, mean_results['f1_macro'])

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

    @staticmethod
    def calculate_confidence_interval(metric: list, mean_metric: float) -> Tuple[float, float]:
        tc = 2.262
        std = calculate_std(np.array(metric))

        ic_lower = mean_metric - tc * (std / math.sqrt(N_SPLITS))
        ic_upper = mean_metric + tc * (std / math.sqrt(N_SPLITS))

        return ic_lower, ic_upper

    @staticmethod
    def get_confusion_matrix_values(y_true: np.ndarray, y_predicted: np.ndarray):
        """Calculates the number of tp, fp, tn, fn for binary datasets"""
        true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0

        for i in range(len(y_true)):
            if y_true[i] == y_predicted[i] == 1:
                true_positive += 1
            if y_predicted[i] == 1 and y_true[i] != y_predicted[i]:
                false_positive += 1
            if y_true[i] == y_predicted[i] == 0:
                true_negative += 1
            if y_predicted[i] == 0 and y_true[i] != y_predicted[i]:
                false_negative += 1

        return true_positive, false_positive, true_negative, false_negative

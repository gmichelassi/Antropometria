import math
import numpy as np
import os
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from abc import ABC, abstractmethod
from antropometria.config.constants import N_SPLITS
from antropometria.utils.metrics import calculate_mean, calculate_std
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Any, Tuple


class ErrorEstimation(ABC):
    def __init__(self, x: np.ndarray, y: np.ndarray, class_count: list[int], estimator: Any, sampling: str):
        self.class_count = class_count
        self.estimator = estimator
        self.sampling = sampling
        self.x = x
        self.y = y

    @abstractmethod
    def run_error_estimation(self) -> dict[str, tuple[float, float]]:
        raise NotImplementedError()

    @abstractmethod
    def __split_dataset(self) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    @abstractmethod
    def __complete_fold(self, x: np.ndarray, y: np.ndarray, current_fold: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def __predict_folds(
            self,
            folds:  list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> Tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
        accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro =\
            [], [], [], [], [], [], []
        for i in range(N_SPLITS):
            self.estimator.fit(folds[i][0], folds[i][1])
            y_predict = self.estimator.predict(folds[i][2])

            accuracy.append(accuracy_score(y_true=folds[i][3], y_pred=y_predict))
            precision_micro.append(precision_score(y_true=folds[i][3], y_pred=y_predict, average='micro'))
            recall_micro.append(recall_score(y_true=folds[i][3], y_pred=y_predict, average='micro'))
            f1_micro.append(f1_score(y_true=folds[i][3], y_pred=y_predict, average='micro'))
            precision_macro.append(precision_score(y_true=folds[i][3], y_pred=y_predict, average='macro'))
            recall_macro.append(recall_score(y_true=folds[i][3], y_pred=y_predict, average='macro'))
            f1_macro.append(f1_score(y_true=folds[i][3], y_pred=y_predict, average='macro'))

        return accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro

    def __calculate_results(
            self,
            accuracy: list[float],
            precision_micro: list[float],
            recall_micro: list[float],
            f1_micro: list[float],
            precision_macro: list[float],
            recall_macro: list[float],
            f1_macro: list[float]
    ) -> dict[str, tuple[float, float]]:
        mean_results = calculate_mean({
            'accuracy': accuracy,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro
        })
        f1_micro_ic = self.__calculate_confidence_interval(f1_micro, mean_results['f1_micro'])
        f1_macro_ic = self.__calculate_confidence_interval(f1_macro, mean_results['f1_macro'])

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
    def __calculate_confidence_interval(metric: list, mean_metric: float) -> Tuple[float, float]:
        tc = 2.262
        std = calculate_std(np.array(metric))

        ic_lower = mean_metric - tc * (std / math.sqrt(N_SPLITS))
        ic_upper = mean_metric + tc * (std / math.sqrt(N_SPLITS))

        return ic_lower, ic_upper

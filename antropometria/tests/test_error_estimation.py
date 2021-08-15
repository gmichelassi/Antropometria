from context import set_tests_context
set_tests_context()

import numpy as np
import pytest

from antropometria.config.constants.general import FIELDNAMES, CV, N_SPLITS
from antropometria.utils.error_estimation.ErrorEstimation import ErrorEstimation
from typing import Any, List, Tuple
from sklearn.datasets import make_classification
from sklearn.svm import SVC

N_FEATURES = 144
N_SAMPLES = 100
X, Y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, weights=[0.4, 0.6])
N_CLASSES, CLASSES_COUNT = np.unique(Y, return_counts=True)


class DummyConcreteErrorEstimation(ErrorEstimation):
    """
    This is a dummy implementation of the abstract class Error Estimation.
    It is used to test the concrete methods.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, class_count: List[int], estimator: Any):
        super(DummyConcreteErrorEstimation, self).__init__(x, y, class_count, estimator)

    def run_error_estimation(self) -> dict[str, Tuple[float, float]]:
        raise NotImplementedError

    def get_folds(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        raise NotImplementedError


class TestErrorEstimation:
    @pytest.fixture
    def error_estimation(self):
        return DummyConcreteErrorEstimation(X, Y, CLASSES_COUNT, SVC())

    @pytest.fixture
    def generate_folds(self):
        folds = []
        for train_index, test_index in CV.split(X, Y):
            x_train: np.ndarray = X[train_index]
            y_train: np.ndarray = Y[train_index]
            x_test: np.ndarray = X[test_index]
            y_test: np.ndarray = Y[test_index]
            folds.append((x_train, y_train, x_test, y_test))

        return folds

    def test_abstract_methods(self, error_estimation):
        with pytest.raises(NotImplementedError):
            error_estimation.run_error_estimation()
            error_estimation.get_folds()

    def test_calculate_metrics(self, error_estimation, generate_folds):
        accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro\
            = error_estimation.calculate_metrics(generate_folds)

        for i in range(N_SPLITS):
            assert isinstance(accuracy[i], float)
            assert isinstance(precision_micro[i], float)
            assert isinstance(recall_micro[i], float)
            assert isinstance(f1_micro[i], float)
            assert isinstance(precision_macro[i], float)
            assert isinstance(recall_macro[i], float)
            assert isinstance(f1_macro[i], float)

    def test_calculate_results(self, error_estimation):
        result_dict = error_estimation.calculate_results(
            accuracy=[0.99, 0.98, 0.95],
            precision_micro=[0.99, 0.98, 0.95],
            recall_micro=[0.99, 0.98, 0.95],
            f1_micro=[0.99, 0.98, 0.95],
            precision_macro=[0.99, 0.98, 0.95],
            recall_macro=[0.99, 0.98, 0.95],
            f1_macro=[0.99, 0.98, 0.95]
        )

        assert list(result_dict.keys()) == FIELDNAMES[10:len(FIELDNAMES) - 1]
        assert isinstance(result_dict['err_accuracy'], float)
        assert isinstance(result_dict['err_precision_micro'], float)
        assert isinstance(result_dict['err_recall_micro'], float)
        assert isinstance(result_dict['err_f1score_micro'], float)
        assert isinstance(result_dict['err_f1micro_ic'], tuple)
        assert isinstance(result_dict['err_f1micro_ic'][0], float)
        assert isinstance(result_dict['err_f1micro_ic'][1], float)
        assert isinstance(result_dict['err_precision_macro'], float)
        assert isinstance(result_dict['err_recall_macro'], float)
        assert isinstance(result_dict['err_f1score_macro'], float)
        assert isinstance(result_dict['err_f1micro_ic'], tuple)
        assert isinstance(result_dict['err_f1macro_ic'][0], float)
        assert isinstance(result_dict['err_f1macro_ic'][1], float)

    def test_calculate_confidence_interval(self, error_estimation):
        accuracy = [0.71, 0.74, 0.72]
        mean_accuracy = float(sum(accuracy)) / max(len(accuracy), 1)
        ic_lower, ic_upper = error_estimation.calculate_confidence_interval(accuracy, mean_accuracy)

        assert ic_lower <= ic_upper
        assert isinstance(ic_lower, float)
        assert isinstance(ic_upper, float)

    def test_get_confusion_matrix_values(self, error_estimation):
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
        y_predicted = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        tp, fp, tn, fn = error_estimation.get_confusion_matrix_values(y_true, y_predicted)

        assert tp + fp + tn + fn == len(y_true)
        assert tp == 3
        assert fp == 2
        assert tn == 3
        assert fn == 2

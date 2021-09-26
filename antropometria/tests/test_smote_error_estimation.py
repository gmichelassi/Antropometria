from context import set_tests_context
set_tests_context()

import numpy as np
import pytest

from antropometria.config.constants.general import CV, N_SPLITS
from antropometria.utils.error_estimation.SmoteErrorEstimation import SmoteErrorEstimation
from pytest import approx
from sklearn.datasets import make_classification
from sklearn.svm import SVC

N_FEATURES = 144
N_SAMPLES = 100
X, Y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, weights=[0.4, 0.6])
N_CLASSES, CLASSES_COUNT = np.unique(Y, return_counts=True)


class TestSmoteErrorEstimation:
    @pytest.fixture
    def error_estimation(self):
        return SmoteErrorEstimation(X, Y, CLASSES_COUNT, SVC())

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

    def test_complete_fold(self, error_estimation, generate_folds):
        current_fold = 0
        for x_train, y_train, x_test, y_test in generate_folds:
            x, y = error_estimation.complete_fold(x_train, y_train, current_fold)

            count = sum(len(error_estimation.splitted_synthetic_x[i]) for i in range(N_SPLITS))

            assert x.shape[1] == x_train.shape[1]
            assert x.shape[0] >= x_train.shape[0]
            assert y.shape >= y_train.shape
            assert approx(x.shape[0], rel=1) == x_train.shape[0] + count

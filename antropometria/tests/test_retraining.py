import pytest

from context import set_tests_context
set_tests_context()

import numpy as np

from antropometria.classifiers.RandomForests import RandomForests
from antropometria.utils.training.retraining import complete_fold, error_estimation, N_SPLITS
from antropometria.config.constants import EMPTY_ERROR_ESTIMATION_DICT
from pytest import approx
from random import randrange
from sklearn.datasets import make_classification


ORIGINAL_X = np.full((10, 5), 1)
ORIGINAL_Y = np.ones((10,))
SYNTHETIC_X = np.full((20, 5), 1)
SYNTHETIC_Y = np.ones((20,))
CURRENT_FOLD = randrange(10)

N_FEATURES = 144
N_SAMPLES = 100
X_BALANCED_DATA, Y_BALANCED_DATA = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES)

X_UNBALANCED_DATA, Y_UNBALANCED_DATA =\
    make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, weights=[0.4, 0.6])
N_UNBALANCED_CLASSES, UNBALANCED_CLASS_COUNT = np.unique(Y_UNBALANCED_DATA, return_counts=True)

DEFAULT_ERROR_DICT = EMPTY_ERROR_ESTIMATION_DICT
FIELDNAMES = list(EMPTY_ERROR_ESTIMATION_DICT.keys())


class TestRetraining:
    def test_complete_frame_works(self):
        x, y = complete_fold(ORIGINAL_X, ORIGINAL_Y, SYNTHETIC_X, SYNTHETIC_Y, CURRENT_FOLD)
        splitted_synthetic = np.array_split(SYNTHETIC_X, N_SPLITS)

        count = 0
        for i in range(N_SPLITS - 1):
            count += len(splitted_synthetic[i])

        assert x.shape[1] == ORIGINAL_X.shape[1]
        assert x.shape[0] >= ORIGINAL_X.shape[0]

        assert y.shape >= ORIGINAL_Y.shape

        assert approx(x.shape[0], rel=1) == ORIGINAL_X.shape[0] + count

    def test_error_estimation_works(self):
        classifier = RandomForests(n_features=X_UNBALANCED_DATA.shape[1])
        result_dict = error_estimation(
            X_UNBALANCED_DATA, Y_UNBALANCED_DATA, UNBALANCED_CLASS_COUNT, classifier.estimator
        )

        assert list(result_dict.keys()) == FIELDNAMES

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

    def test_error_estimation_with_three_classes(self):
        with pytest.raises(ValueError):
            error_estimation(X_BALANCED_DATA, Y_BALANCED_DATA, ['one', 'two', 'three'], None)

    def test_error_estimation_with_one_class(self):
        with pytest.raises(ValueError):
            error_estimation(X_BALANCED_DATA, Y_BALANCED_DATA, ['ONE CLASS'], None)

    def test_error_estimation_with_balanced_dataset(self):
        result_dict = error_estimation(X_BALANCED_DATA, Y_BALANCED_DATA, [50, 50], None)

        assert result_dict == DEFAULT_ERROR_DICT

from context import set_tests_context
set_tests_context()

import numpy as np

from antropometria.sampling.OverSampling import OverSampling
from antropometria.sampling.UnderSampling import UnderSampling
from pytest import approx
from sklearn.datasets import make_classification


N_FEATURES = 144
N_SAMPLES = 100
X_DATA, Y_DATA = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, weights=[0.4, 0.6])
N_CLASSES, CLASSES_COUNT = np.unique(Y_DATA, return_counts=True)
difference_of_classes = abs(CLASSES_COUNT[0] - CLASSES_COUNT[1])


def unique(y_balanced: np.ndarray):
    return np.unique(y_balanced, return_counts=True)


class TestDataSampling:
    def test_oversampling_with_default_smote(self):
        x_balanced, y_balanced = OverSampling('default').fit_transform(X_DATA, Y_DATA)
        balanced_n_classes, balanced_classes_count = unique(y_balanced)

        assert np.array_equal(N_CLASSES, balanced_n_classes)
        assert approx(balanced_classes_count[0], rel=5) == balanced_classes_count[1]
        assert CLASSES_COUNT[0] + difference_of_classes == CLASSES_COUNT[1] \
               or CLASSES_COUNT[0] == CLASSES_COUNT[1] + difference_of_classes

    def test_oversampling_with_smote_borderline(self):
        x_balanced, y_balanced = OverSampling('Borderline').fit_transform(X_DATA, Y_DATA)
        balanced_n_classes, balanced_classes_count = unique(y_balanced)

        assert np.array_equal(N_CLASSES, balanced_n_classes)
        assert approx(balanced_classes_count[0], rel=5) == balanced_classes_count[1]
        assert CLASSES_COUNT[0] + difference_of_classes == CLASSES_COUNT[1] \
               or CLASSES_COUNT[0] == CLASSES_COUNT[1] + difference_of_classes

    def test_oversampling_with_smote_kmeans(self):
        x_balanced, y_balanced = OverSampling('KMeans').fit_transform(X_DATA, Y_DATA)
        balanced_n_classes, balanced_classes_count = unique(y_balanced)

        assert np.array_equal(N_CLASSES, balanced_n_classes)
        assert approx(balanced_classes_count[0], rel=5) == balanced_classes_count[1]
        assert CLASSES_COUNT[0] + difference_of_classes == CLASSES_COUNT[1] \
               or CLASSES_COUNT[0] == CLASSES_COUNT[1] + difference_of_classes

    def test_oversampling_with_smote_svm(self):
        x_balanced, y_balanced = OverSampling('SVM').fit_transform(X_DATA, Y_DATA)
        balanced_n_classes, balanced_classes_count = unique(y_balanced)

        assert np.array_equal(N_CLASSES, balanced_n_classes)
        assert approx(balanced_classes_count[0], rel=5) == balanced_classes_count[1]
        assert CLASSES_COUNT[0] + difference_of_classes == CLASSES_COUNT[1] \
               or CLASSES_COUNT[0] == CLASSES_COUNT[1] + difference_of_classes

    def test_oversampling_with_smote_tomek(self):
        x_balanced, y_balanced = OverSampling('Tomek').fit_transform(X_DATA, Y_DATA)
        balanced_n_classes, balanced_classes_count = unique(y_balanced)

        assert np.array_equal(N_CLASSES, balanced_n_classes)
        assert approx(balanced_classes_count[0], rel=5) == balanced_classes_count[1]
        assert CLASSES_COUNT[0] + difference_of_classes == CLASSES_COUNT[1] \
               or CLASSES_COUNT[0] == CLASSES_COUNT[1] + difference_of_classes

    def test_random_undersampling(self):
        x_balanced, y_balanced = UnderSampling().fit_transform(X_DATA, Y_DATA)
        balanced_n_classes, balanced_classes_count = unique(y_balanced)

        assert np.array_equal(N_CLASSES, balanced_n_classes)
        assert approx(balanced_classes_count[0], rel=5) == balanced_classes_count[1]
        assert CLASSES_COUNT[0] + difference_of_classes == CLASSES_COUNT[1] \
               or CLASSES_COUNT[0] == CLASSES_COUNT[1] + difference_of_classes

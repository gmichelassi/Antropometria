import numpy as np
from sklearn.datasets import make_classification

from antropometria.feature_selectors.get_feature_selector import \
    get_feature_selector

N_FEATURES = 144
N_SAMPLES = 100
X_DATA, Y_DATA = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES)


class TestFeatureSelectors:
    def test_pca_works(self):
        feature_selector = get_feature_selector('PCA', int(np.sqrt(N_FEATURES)), N_SAMPLES, N_FEATURES)

        x_reduced = feature_selector.fit_transform(X_DATA, Y_DATA)

        assert x_reduced.shape[1] < X_DATA.shape[1]
        assert x_reduced.shape == (N_SAMPLES, int(np.sqrt(N_FEATURES)))

    def test_mrmr_works(self):
        feature_selector = get_feature_selector('mRMR', int(np.sqrt(N_FEATURES)), N_SAMPLES, N_FEATURES)

        x_reduced = feature_selector.fit_transform(X_DATA, Y_DATA)

        assert x_reduced.shape[1] < X_DATA.shape[1]
        assert x_reduced.shape == (N_SAMPLES, int(np.sqrt(N_FEATURES)))

    def test_fcbf_works(self):
        feature_selector = get_feature_selector('FCBF', int(np.sqrt(N_FEATURES)), N_SAMPLES, N_FEATURES)

        x_reduced = feature_selector.fit_transform(X_DATA, Y_DATA)

        assert x_reduced.shape[1] < X_DATA.shape[1]
        assert x_reduced.shape == (N_SAMPLES, int(np.sqrt(N_FEATURES)))

    def test_cfs_works(self):
        feature_selector = get_feature_selector('CFS', int(np.sqrt(N_FEATURES)), N_SAMPLES, N_FEATURES)

        x_reduced = feature_selector.fit_transform(X_DATA, Y_DATA)

        assert x_reduced.shape[1] < X_DATA.shape[1]
        assert x_reduced.shape == (N_SAMPLES, int(np.sqrt(N_FEATURES)))

    def test_rfs_works(self):
        feature_selector = get_feature_selector('RFS', int(np.sqrt(N_FEATURES)), N_SAMPLES, N_FEATURES)

        x_reduced = feature_selector.fit_transform(X_DATA, Y_DATA)

        assert x_reduced.shape[1] < X_DATA.shape[1]
        assert x_reduced.shape == (N_SAMPLES, int(np.sqrt(N_FEATURES)))

    def test_relieff_works(self):
        feature_selector = get_feature_selector('ReliefF', int(np.sqrt(N_FEATURES)), N_SAMPLES, N_FEATURES)

        x_reduced = feature_selector.fit_transform(X_DATA, Y_DATA)

        assert x_reduced.shape[1] < X_DATA.shape[1]
        assert x_reduced.shape == (N_SAMPLES, int(np.sqrt(N_FEATURES)))

    def test_rfselect_works(self):
        feature_selector = get_feature_selector('RFSelect', int(np.sqrt(N_FEATURES)), N_SAMPLES, N_FEATURES)

        x_reduced = feature_selector.fit_transform(X_DATA, Y_DATA)

        assert x_reduced.shape[1] < X_DATA.shape[1]

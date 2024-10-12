from typing import Union

from sklearn.decomposition import PCA
from skrebate import ReliefF

from antropometria.config.types import Reduction
from antropometria.feature_selectors.CFS import CFS
from antropometria.feature_selectors.FCBF import FCBF
from antropometria.feature_selectors.MRMR import MRMR
from antropometria.feature_selectors.RFS import RFS
from antropometria.feature_selectors.RFSelect import RFSelect

FeatureSelector = Union[CFS, FCBF, MRMR, RFS, RFSelect, PCA, ReliefF]

REDUCTIONS = {
    'pca': PCA,
    'mrmr': MRMR,
    'fcbf': FCBF,
    'cfs': CFS,
    'rfs': RFS,
    'relieff': ReliefF,
    'rfselect': RFSelect,
}


def get_feature_selector(
        reduction: Reduction,
        n_features_to_keep: int,
        instances: int,
        features: int
) -> FeatureSelector:
    lowered_reduction = reduction.lower()
    reduction_method = REDUCTIONS[lowered_reduction]

    if lowered_reduction == 'pca':
        if n_features_to_keep < min(instances, features):
            return reduction_method(n_components=n_features_to_keep)
        return reduction_method(n_components=min(instances, features))

    if lowered_reduction == 'relieff':
        return reduction_method(n_features_to_select=n_features_to_keep, n_neighbors=100, n_jobs=-1)

    return reduction_method(n_features_to_keep)

from antropometria.feature_selectors.CFS import CFS
from antropometria.feature_selectors.FCBF import FCBF
from antropometria.feature_selectors.MRMR import MRMR
from antropometria.feature_selectors.RFS import RFS
from antropometria.feature_selectors.RFSelect import RFSelect
from sklearn.decomposition import PCA
from skrebate import ReliefF
from typing import Union


def get_feature_selector(
        reduction: str,
        n_features_to_keep: int,
        instances: int,
        features: int
) -> Union[CFS, FCBF, MRMR, RFS, RFSelect, PCA, ReliefF]:
    if reduction == 'PCA':
        if n_features_to_keep < min(instances, features):
            return PCA(n_components=n_features_to_keep)
        else:
            return PCA(n_components=min(instances, features))
    elif reduction == 'mRMR':
        return MRMR(n_features_to_select=n_features_to_keep)
    elif reduction == 'FCBF':
        return FCBF(n_features_to_select=n_features_to_keep)
    elif reduction == 'CFS':
        return CFS(n_features_to_select=n_features_to_keep)
    elif reduction == 'RFS':
        return RFS(n_features_to_select=n_features_to_keep)
    elif reduction == 'ReliefF':
        return ReliefF(n_features_to_select=n_features_to_keep, n_neighbors=100, n_jobs=-1)
    elif reduction == 'RFSelect':
        return RFSelect()

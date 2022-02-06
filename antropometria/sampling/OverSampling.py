import numpy as np
import pandas as pd

from antropometria.config.types import Sampling
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from sklearn.cluster import KMeans
from typing import Tuple, Union

RANDOM_STATE = 10000
DEFAULT = 'default'


class OverSampling:
    def __init__(self, algorithm: Union[Sampling, DEFAULT] = 'default'):
        self.algorithm = algorithm

    def __get_smote(self) -> Union[SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, SMOTETomek]:
        if self.algorithm == 'Borderline':
            return BorderlineSMOTE(random_state=RANDOM_STATE)
        if self.algorithm == 'KMeans':
            return KMeansSMOTE(random_state=RANDOM_STATE, kmeans_estimator=KMeans(n_clusters=20))
        if self.algorithm == 'SVM':
            return SVMSMOTE(random_state=RANDOM_STATE)
        if self.algorithm == 'Tomek':
            return SMOTETomek(random_state=RANDOM_STATE)

        return SMOTE(random_state=RANDOM_STATE)

    def fit_transform(self, x: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        smote = self.__get_smote()

        x_novo, y_novo = smote.fit_resample(x, y)

        return x_novo, y_novo

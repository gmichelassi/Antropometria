import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek

RANDOM_STATE = 10000


class OverSampling:
    def __init__(self, algorithm: str = 'default'):
        self.algorithm = algorithm

    def __get_smote(self):
        if self.algorithm == 'Borderline':
            return BorderlineSMOTE(random_state=RANDOM_STATE)
        elif self.algorithm == 'KMeans':
            return KMeansSMOTE(random_state=RANDOM_STATE, kmeans_estimator=KMeans(n_clusters=20))
        elif self.algorithm == 'SVM':
            return SVMSMOTE(random_state=RANDOM_STATE)
        elif self.algorithm == 'Tomek':
            return SMOTETomek(random_state=RANDOM_STATE)

        return SMOTE(random_state=RANDOM_STATE)

    def fit_transform(self, x: pd.DataFrame, y: np.array):
        smote = self.__get_smote()

        x_novo, y_novo = smote.fit_resample(x, y)

        return x_novo, y_novo
import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

RANDOM_STATE = 10000


class UnderSampling:
    @staticmethod
    def fit_transform(x: pd.DataFrame, y: np.array):
        x_novo, y_novo = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(x, y)
        return x_novo, y_novo

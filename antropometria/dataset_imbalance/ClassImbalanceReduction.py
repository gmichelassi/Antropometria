from .OverSampling import OverSampling
from .UnderSampling import UnderSampling

import pandas as pd
import numpy as np


class ClassImbalanceReduction:
    def __init__(self, dataset: pd.DataFrame, labels: np.ndarray, algorithm: str):
        self.dataset = dataset
        self.labels = labels
        self.algorithm = algorithm

    def apply_class_imbalance_reduction(self) -> tuple[pd.DataFrame, np.ndarray]:
        if self.algorithm == 'Random':
            return UnderSampling().fit_transform(self.dataset, self.labels)

        return OverSampling(self.algorithm).fit_transform(self.dataset, self.labels)

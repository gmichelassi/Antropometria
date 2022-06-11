import numpy as np
import pandas as pd

from antropometria.config.constants import TEMPORARY_RANDOM_SAMPLES, TEMPORARY_RANDOM_SAMPLES_LABELS
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple

RANDOM_STATE = 10000


class UnderSampling:
    @staticmethod
    def fit_transform(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        under_sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        x_novo, y_novo = under_sampler.fit_resample(x, y)

        array_with_removed_samples = np.delete(x, under_sampler.sample_indices_, axis=0)
        removed_labels = np.delete(y, under_sampler.sample_indices_, axis=0)
        pd.DataFrame(array_with_removed_samples).to_csv(TEMPORARY_RANDOM_SAMPLES, index=False)
        pd.DataFrame(removed_labels).to_csv(TEMPORARY_RANDOM_SAMPLES_LABELS, index=False)

        return x_novo, y_novo

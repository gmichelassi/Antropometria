from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from antropometria.config.constants import (TEMPORARY_RANDOM_SAMPLES,
                                            TEMPORARY_RANDOM_SAMPLES_LABELS)

RANDOM_STATE = 10000


class UnderSampling:
    @staticmethod
    def fit_transform(x: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        under_sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        x_novo, y_novo = under_sampler.fit_resample(x, y)

        array_with_removed_samples = np.delete(x, under_sampler.sample_indices_, axis=0)
        removed_labels = np.delete(y, under_sampler.sample_indices_, axis=0)
        pd.DataFrame(array_with_removed_samples).to_csv(TEMPORARY_RANDOM_SAMPLES, index=False)
        pd.DataFrame(removed_labels).to_csv(TEMPORARY_RANDOM_SAMPLES_LABELS, index=False)

        return x_novo, y_novo

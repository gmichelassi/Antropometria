import numpy as np
import pandas as pd

from antropometria.config.constants import PROCESSED_DIR
from antropometria.config.types import Reduction, Sampling
from typing import Optional, Tuple


def load_processed_data(
    name: str,
    apply_min_max: bool,
    p_filter: float,
    reduction: Optional[Reduction],
    sampling: Optional[Sampling],
) -> Tuple[np.ndarray, np.ndarray]:
    preprocessing_directory = f'{reduction}_{sampling}_{p_filter}_{apply_min_max}'
    directory = PROCESSED_DIR + preprocessing_directory

    x = pd.read_csv(f'{directory}/{name}_data.csv', index_col=False, header=None)
    y = pd.read_csv(f'{directory}/{name}_labels.csv', index_col=False, header=None)

    return x.to_numpy(), y.T.to_numpy().flatten()

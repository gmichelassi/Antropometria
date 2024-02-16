import numpy as np
import os

import pandas as pd

from .deprecated_preprocess import deprecated_preprocess
from antropometria.config import FILTERS, MIN_MAX_NORMALIZATION, REDUCTIONS, SAMPLINGS
from antropometria.config.constants import PROCESSED_DIR
from itertools import product
from pandas import DataFrame
from typing import Tuple


def run_preprocessing(data: Tuple[DataFrame, np.ndarray], name: str):
    setup_directory(PROCESSED_DIR)

    for reduction, sampling, p_filter, apply_min_max in product(REDUCTIONS, SAMPLINGS, FILTERS, MIN_MAX_NORMALIZATION):
        preprocessing_directory = f'{reduction}_{sampling}_{p_filter}_{apply_min_max}'
        output_directory = PROCESSED_DIR + preprocessing_directory

        setup_directory(output_directory)

        x, y = deprecated_preprocess(data, apply_min_max, p_filter, reduction, sampling)

        pd.DataFrame(x).to_csv(f'{output_directory}/{name}_data.csv', index=False, header=False)
        pd.DataFrame(y).to_csv(f'{output_directory}/{name}_labels.csv', index=False, header=False)


def setup_directory(directory: str):
    if not os.path.exists(directory):
        os.mkdir(directory)

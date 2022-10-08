import numpy as np
import os

from .preprocess import preprocess
from antropometria.config import FILTERS, MIN_MAX_NORMALIZATION, REDUCTIONS, SAMPLINGS
from itertools import product
from pandas import DataFrame
from typing import Tuple


PREPROCESSING_PARAMS = product(REDUCTIONS, SAMPLINGS, FILTERS, MIN_MAX_NORMALIZATION)
PROCESSED_DIR = 'antropometria/data/processed/'


def run_preprocessing(data: Tuple[DataFrame, np.ndarray]):
    setup_processed_directory()

    for reduction, sampling, p_filter, apply_min_max in PREPROCESSING_PARAMS:
        x, y = preprocess(data, apply_min_max, p_filter, reduction, sampling)


def setup_processed_directory():
    if not os.path.exists(PROCESSED_DIR):
        os.mkdir(PROCESSED_DIR)

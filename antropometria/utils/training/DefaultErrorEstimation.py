import numpy as np
import os
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ErrorEstimation import ErrorEstimation
from typing import Any, Tuple


class DefaultErrorEstimation(ErrorEstimation):
    def __init__(self, x: np.ndarray, y: np.ndarray, class_count: list[int], estimator: Any, sampling: str):
        super(DefaultErrorEstimation, self).__init__(x, y, class_count, estimator, sampling)

    def run_error_estimation(self) -> dict[str, tuple[float, float]]:
        pass

    def __split_dataset(self) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        pass

    def __complete_fold(self, x: np.ndarray, y: np.ndarray, current_fold: int) -> Tuple[np.ndarray, np.ndarray]:
        pass
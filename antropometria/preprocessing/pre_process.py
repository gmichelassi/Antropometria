import os
import platform
import time
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from antropometria.config import (FILTERS, MIN_MAX_NORMALIZATION,
                                  PROCESSED_DIR, REDUCTIONS, SAMPLINGS, logger)
from antropometria.dataset_imbalance import ClassImbalanceReduction
from antropometria.feature_selectors.get_feature_selector import \
    get_feature_selector
from antropometria.statistics import PeasonCorrelationFeatureSelector
from antropometria.utils.timeout import timeout

from .calculate_number_of_features_to_keep import \
    calculate_number_of_n_features_to_keep

log = logger.get_logger(__file__)


class PreProcess:
    def __init__(
        self,
        dataset: pd.DataFrame,
        labels: np.ndarray,
        name: str,
        reductions: Optional[list[str]] = REDUCTIONS,
        samplings: Optional[list[str]] = SAMPLINGS,
        filters: Optional[list[float]] = FILTERS,
        min_max_normalization: Optional[list[bool]] = MIN_MAX_NORMALIZATION
    ):
        self.dataset = dataset
        self.labels = labels
        self.name = name
        self.reductions = reductions
        self.samplings = samplings
        self.filters = filters
        self.min_max_normalization = min_max_normalization

        self.__setup_output_directory(PROCESSED_DIR)

    @timeout(seconds=7500, use_timeout=(platform.system().lower() != 'windows'))
    def run(self):
        log.info(f'Preprocessing {self.name} dataset')
        preprocessing_initial_time = time.time()

        for reduction, sampling, p_filter, apply_min_max in self.__combinations():
            preprocessing_directory = f'{reduction}_{sampling}_{p_filter}_{apply_min_max}'
            output_directory = PROCESSED_DIR + preprocessing_directory

            self.__setup_output_directory(output_directory)

            x, y = self.run_individual(apply_min_max, p_filter, reduction, sampling)

            pd.DataFrame(x).to_csv(f'{output_directory}/{self.name}_data.csv', index=False, header=False)
            pd.DataFrame(y).to_csv(f'{output_directory}/{self.name}_labels.csv', index=False, header=False)

        log.info(f"Preprocessing all combinations took {(time.time() - preprocessing_initial_time) / 60} minutes")

    def run_individual(
        self,
        apply_min_max: Optional[bool],
        p_filter: Optional[float],
        reduction: Optional[str],
        sampling: Optional[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        log.info(f'Preprocessing data with [{reduction}, {sampling}, filter {p_filter}, min_max {apply_min_max}]')
        preprocessing_initial_time = time.time()

        x, y = self.dataset, self.labels

        if 0.0 < p_filter <= 0.99:
            log.info('Applying pearson correlation filter')
            x = self.apply_pearson_correlation_filter(x, p_filter)

        if apply_min_max:
            log.info('Applying min max normalization')
            x = self.apply_min_max_normalization(x)

        _, current_number_of_features = x.shape

        n_features_to_keep = calculate_number_of_n_features_to_keep(
            self.__original_number_of_features(), current_number_of_features, dataset=x
        )

        if reduction is not None:
            log.info(f'Applying {reduction} reduction')
            x = self.apply_feature_selection(
                x, y, reduction, n_features_to_keep, self.__instances(), current_number_of_features
            )

        if sampling is not None:
            log.info(f'Applying {sampling} sampling')
            x, y = self.apply_class_imbalance_reduction(x, y, sampling)

        log.info(f"Dataset shape after pre processing: {x.shape}")

        log.info(f"Preprocessing took {(time.time() - preprocessing_initial_time) / 60} minutes")

        return x, y

    @staticmethod
    def apply_min_max_normalization(dataset: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(MinMaxScaler().fit_transform(dataset.copy()), columns=dataset.columns)

    @staticmethod
    def apply_pearson_correlation_filter(dataset: pd.DataFrame, p_filter: float) -> pd.DataFrame:
        return PeasonCorrelationFeatureSelector(dataset.copy(), p_filter).apply_pearson_feature_selection()

    @staticmethod
    def apply_feature_selection(
        dataset: pd.DataFrame,
        labels: np.ndarray,
        reduction: str,
        n_features_to_keep: int,
        instances: int,
        current_number_of_features: int
    ) -> np.ndarray:
        feature_selector = get_feature_selector(reduction, n_features_to_keep, instances, current_number_of_features)
        return feature_selector.fit_transform(dataset.to_numpy(), labels)

    @staticmethod
    def apply_class_imbalance_reduction(
        dataset: pd.DataFrame, labels: np.ndarray, sampling: str
    ) -> tuple[pd.DataFrame, np.ndarray]:
        return ClassImbalanceReduction(dataset, labels, sampling).apply_class_imbalance_reduction()

    def __instances(self):
        return self.dataset.shape[0]

    def __original_number_of_features(self):
        return self.dataset.shape[1]

    def __class_counts(self):
        return np.unique(self.labels, return_counts=True)[1]

    def __combinations(self):
        return product(self.reductions, self.samplings, self.filters, self.min_max_normalization)

    @staticmethod
    def __setup_output_directory(directory: str):
        if not os.path.exists(directory):
            os.mkdir(directory)

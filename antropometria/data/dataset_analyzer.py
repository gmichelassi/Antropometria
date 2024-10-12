import numpy as np
import pandas as pd


class DatasetAnalyzer:
    def __init__(self, dataset: pd.DataFrame, labels: np.array):
        self.dataset = dataset
        self.labels = labels

        if len(self.dataset) != len(self.labels):
            raise ValueError('The number of instances in the dataset and the number of labels should be the same.')

    def feature_names(self) -> list[str]:
        return self.dataset.columns.tolist()

    def class_names(self) -> list[str]:
        return np.unique(self.labels).tolist()

    def number_of_features(self) -> int:
        return len(self.feature_names())

    def number_of_instances(self) -> int:
        return len(self.dataset)

    def number_of_classes(self) -> int:
        return len(np.unique(self.labels))

    def mean_of(self, feature: str) -> float:
        if feature not in self.feature_names():
            raise ValueError(f'Feature {feature} not found in the given dataset.')

        return self.dataset[feature].mean()

    def median_of(self, feature: str) -> float:
        if feature not in self.feature_names():
            raise ValueError(f'Feature {feature} not found in the given dataset.')

        return self.dataset[feature].median()

    def std_of(self, feature: str) -> float:
        if feature not in self.feature_names():
            raise ValueError(f'Feature {feature} not found in the given dataset.')

        return self.dataset[feature].std()

    def min_of(self, feature: str):
        if feature not in self.feature_names():
            raise ValueError(f'Feature {feature} not found in the given dataset.')

        return self.dataset[feature].min()

    def max_of(self, feature: str):
        if feature not in self.feature_names():
            raise ValueError(f'Feature {feature} not found in the given dataset.')

        return self.dataset[feature].max()

    def count_of(self, feature: str) -> int:
        if feature not in self.feature_names():
            raise ValueError(f'Feature {feature} not found in the given dataset.')

        return self.dataset[feature].count()

    def describe(self) -> pd.DataFrame:
        return self.dataset.describe()

    def class_distribution(self) -> dict[str, int]:
        class_names, count = np.unique(self.labels, return_counts=True)

        return dict(zip(class_names, count))

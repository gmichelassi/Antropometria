import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from antropometria.exceptions import MissingDatasetError

BASE_PATH = './antropometria/data'
LABEL_REGEX = '.*(label).*'
RANDOM_STATE = 10000


class DatasetReader:
    """Class responsible for reading a csv dataset."""
    def __init__(
        self,
        folder: str,
        dataset_name: str,
        classes: Optional[list[str]] = None,
        columns_to_drop: list[str] = None
    ):
        self.folder = folder
        self.dataset_name = dataset_name
        self.classes = classes
        self.columns_to_drop = columns_to_drop

    def read(self) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
        """Read the dataset from a csv file.

        Returns:
            - pd.DataFrame - The dataset.
            - np.ndarray - The labels.
            - list[int] - The number of instances of each class.
        """
        if self.__is_multiple_files():
            x, y = self.__read_data_from_multiple_files()
        else:
            x, y = self.__read_data_from_single_file()

        x, y = shuffle(x, y, random_state=RANDOM_STATE)

        _, classes_count = np.unique(y, return_counts=True)

        return x, y, classes_count.tolist()

    def __read_data_from_multiple_files(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Read the dataset from multiple files.

        Returns:
            - pd.DataFrame - The dataset.
            - np.ndarray - The labels.
        """
        x = pd.DataFrame()
        y = np.array([])

        for class_name in self.classes:
            data, labels = self.__read_data_from_single_file(class_name=class_name)

            x = pd.concat([x, data])
            y = np.concatenate((y, labels))

        return x, y

    def __read_data_from_single_file(self, class_name: Optional[str] = None) -> tuple[pd.DataFrame, np.ndarray]:
        """Read the dataset from a single file.

        Args:
            class_name: str (optional) - The name of the class to be used in the file name.

        Returns:
            - pd.DataFrame - The dataset.
            - np.ndarray - The labels.
        """
        filepath = self.__filepath(class_name)

        if not os.path.isfile(filepath):
            raise MissingDatasetError(folder=self.folder, name=self.dataset_name)

        dataset = pd.read_csv(filepath_or_buffer=filepath)

        label_columns = dataset.filter(regex=LABEL_REGEX).columns
        labels = dataset[label_columns[0]].to_numpy() if len(label_columns) > 0 else np.array([class_name] * len(dataset))

        if self.columns_to_drop is not None and len(self.columns_to_drop) > 0:
            return dataset.drop(labels=self.columns_to_drop, axis=1), labels

        return dataset, labels

    def __filepath(self, class_name: Optional[str] = None) -> str:
        """Get the file path of the dataset.

        Args:
            class_name: str (optional) - The name of the class to be used in the file name.

        Returns:
            str - The file path of the dataset.
        """
        if class_name is None:
            return f'{BASE_PATH}/{self.folder}/{self.dataset_name}.csv'

        return f'{BASE_PATH}/{self.folder}/{class_name}_{self.dataset_name}.csv'

    def __is_multiple_files(self) -> bool:
        """
        Check if the dataset is stored in multiple files (by checking if it can find all files with the expected path)

        Returns:
            bool - True if the dataset is stored in multiple files, False otherwise.
        """
        if self.classes is None or len(self.classes) in [0, 1] or os.path.isfile(self.__filepath()):
            return False

        return self.__all_files_exists()

    def __all_files_exists(self) -> bool:
        """
        Check if all files exists

        Returns:
            bool - True if all files exists, False otherwise.
        """
        for class_name in self.classes:
            if not os.path.isfile(self.__filepath(class_name)):
                return False

        return True

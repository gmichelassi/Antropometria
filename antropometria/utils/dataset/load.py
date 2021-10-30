import numpy as np
import os
import pandas as pd

from antropometria.config.exceptions.dataset import MissingDatasetError
from sklearn.utils import shuffle
from typing import Tuple


class LoadData:
    def __init__(self, folder: str, dataset_name: str, classes: list):
        self.folder = folder
        self.dataset_name = dataset_name
        self.classes = classes
        self.LABEL_COLUMN = 'class_label'
        self.LABEL_REGEX = '.*(label).*'
        self.RANDOM_STATE = 10000

    def load(self) -> Tuple[pd.DataFrame, np.ndarray]:
        if len(self.classes) == 0:
            raise IOError('It is not possible to load a dataset with {0} argument'.format(self.classes))

        if len(self.classes) == 1:
            return self.__load_data_in_single_file()

        return self.__load_data_in_multiple_files()

    def __load_data_in_multiple_files(self) -> Tuple[pd.DataFrame, np.ndarray]:
        x = pd.DataFrame()
        y = np.array([])

        label_count = 0
        for class_name in self.classes:
            file_name = f'./antropometria/data/{self.folder}/{class_name}_{self.dataset_name}.csv'

            if not os.path.isfile(file_name):
                raise MissingDatasetError(folder=self.folder, name=self.dataset_name)

            data = self.__load_data_from_file(file_name=file_name)
            label = label_count * np.ones(len(data), dtype=np.int64)

            x = pd.concat([x, data])
            y = np.concatenate((y, label))

            label_count += 1

        x, y = shuffle(x, y, random_state=self.RANDOM_STATE)
        return x, y.astype('int64')

    def __load_data_from_file(self, file_name: str):
        if 'ratio' in self.folder:
            return pd.concat(list(map(lambda x: x, pd.read_csv(file_name, chuncksize=10, dtype=np.float64))))

        dataset = pd.read_csv(file_name)

        if self.folder in ['dlibHOG', 'dlibCNN', 'openCvDNN', 'openCvHaar', 'openFace']:
            return dataset.drop(['img_name', 'id'], axis=1)

        return dataset

    def __load_data_in_single_file(self) -> Tuple[pd.DataFrame, np.ndarray]:
        path = f"./antropometria/data/{self.folder}/{self.dataset_name}.csv"

        if not os.path.isfile(path):
            raise MissingDatasetError(folder=self.folder, name=self.dataset_name)

        data = pd.read_csv(path)

        try:
            labels = data[self.LABEL_COLUMN].values
            data = data.drop(self.LABEL_COLUMN, axis=1)
        except KeyError:
            columns_regex = data.filter(regex='.*(label).*').columns
            if columns_regex.shape[0] != 1:
                raise IOError(f"File do not have columns like '{self.LABEL_COLUMN}' or '{self.LABEL_REGEX}'")

            labels = data[columns_regex].T.values[0]
            data = data.drop(columns_regex, axis=1)

        return data, labels.astype('int64')

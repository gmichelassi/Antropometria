import numpy as np
import pandas as pd
import os

from sklearn.utils import shuffle


class LoadData:
    def __init__(self, folder: str, dataset_name: str, classes: list):
        self.folder = folder
        self.dataset_name = dataset_name
        self.classes = classes
        self.LABEL_COLUMN = 'class_label'
        self.RANDOM_STATE = 10000

    def load(self) -> (pd.DataFrame, np.array):
        if len(self.classes) == 0:
            raise IOError('It is not possible to load a dataset with {0} argument'.format(self.classes))

        if len(self.classes) == 1:
            return self.__load_data_in_single_file()

        return self.__load_data_in_multiple_files()

    def __load_data_in_multiple_files(self) -> (pd.DataFrame, np.array):
        x = pd.DataFrame()
        y = np.array([])

        label_count = 0
        for class_name in self.classes:
            file_name = f'./antropometria/data/{self.folder}/{class_name}_{self.dataset_name}.csv'

            if os.path.isfile(file_name):
                if 'ratio' in self.folder:
                    chuncksize, chunklist = 10, []
                    for chunk in pd.read_csv(file_name, chunksize=chuncksize, dtype=np.float64):
                        chunklist.append(chunk)
                    data = pd.concat(chunklist)
                else:
                    data = pd.read_csv(file_name)
                    if 'dlibHOG' in self.folder:
                        data = data.drop(['img_name', 'id'], axis=1)

                label = label_count * np.ones(len(data), dtype=np.int64)

                x = pd.concat([x, data])
                y = np.concatenate((y, label))
            else:
                raise IOError(f'File not found for params {self.folder} and {self.dataset_name}.')
            label_count += 1

        x, y = shuffle(x, y, random_state=self.RANDOM_STATE)
        return x, y.astype('int64')

    def __load_data_in_single_file(self) -> (pd.DataFrame, np.array):
        path = f"./antropometria/data/{self.folder}/{self.dataset_name}.csv"

        if os.path.isfile(path):
            data = pd.read_csv(path)

            labels = data[self.LABEL_COLUMN].values
            data = data.drop(self.LABEL_COLUMN, axis=1)
            return data, labels.astype('int64')

        raise IOError(f'File not found for params {self.folder} and {self.dataset_name}.')

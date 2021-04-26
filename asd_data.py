from config import logger
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle

import initContext as context
context.loadModules()
log = logger.getLogger(__file__)
random_state = 10000


def load_data(folder='dlibHOG', dataset_name='distances_all_px_eu', classes=None, verbose=True):
    if classes is None or len(classes) <= 1:
        raise IOError('It is not possible to load a dataset with {0} argument. Please insert two or more classes names or use method load_all().'.format(classes))

    if verbose:
        log.info("Loading data from csv file")

    X = pd.DataFrame()
    y = np.array([])

    label_count = 0
    for classe in classes:
        file_name = f'./data/{folder}/{classe}_{dataset_name}.csv'
        if verbose:
            log.info(f'[{label_count}] Classe {classe}: {file_name}')

        if os.path.isfile(file_name):
            if 'ratio' in 'folder':
                chuncksize, chunklist = 10, []
                for chunk in pd.read_csv(file_name, chunksize=chuncksize, dtype=np.float64):
                    chunklist.append(chunk)
                data = pd.concat(chunklist)
            else:
                data = pd.read_csv(file_name)
                if 'dlibHOG' in folder:
                    data = data.drop(['img_name', 'id'], axis=1)

            log.info("Classe {0}: {1}".format(classe, data.shape))
            label = label_count * np.ones(len(data), dtype=np.int)

            X = pd.concat([X, data])
            y = np.concatenate((y, label))
        else:
            log.info("File not found for parameters: [{0}, {1}, {2}]".format(folder, dataset_name, classes))

        label_count += 1

    X, y = shuffle(X, y, random_state=random_state)
    return X, y.astype('int64')


def load_all(folder='casos', dataset_name='distances_all_px_eu', label_column='labels'):
    path = f"./data/{folder}/{dataset_name}.csv"

    if os.path.isfile(path):
        data = pd.read_csv(path)

        if label_name is None:
            raise ValueError('It is not possible to load a dataset without a column for its label. If you have one file for each class, try method load_data().')
        else:
            labels = data[label_column].values
            data = data.drop(label_column, axis=1)

        return data, labels
    else:
        raise IOError(f'File not found for params {folder} and {dataset_name}.')

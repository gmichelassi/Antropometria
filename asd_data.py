from config import logger
from config import classpathDir as cdir
import numpy as np
import pandas as pd
import os
import subprocess
from sklearn.utils import shuffle

import initContext as context
context.loadModules()
log = logger.getLogger(__file__)
random_state = 10000


def __checkDimension(X, y):
    return X.shape[0] == y.shape[0]


def merge_frames(dataframe_list):
    return pd.concat(dataframe_list)


def remove_feature(dataframe, feature):
    return dataframe.drop(feature, axis=1)


def load_by_chunks(file_name):
    chuncksize = 10
    chunk_list = []
    for chunk in pd.read_csv(file_name, chunksize=chuncksize, dtype=np.float64):
        chunk_list.append(chunk)
    return merge_frames(chunk_list)


def load_data(lib='dlibHOG', dataset='distances_all_px_eu', classes=None, ratio=False, verbose=True):
    if classes is None or len(classes) == 1:
        raise IOError(f'It is not possible to load a dataset with {classes} argument. Please insert two or more classes names')

    if verbose:
        log.info("Loading data from csv file")

    X = pd.DataFrame()
    y = np.array([])

    label_count = 0
    for classe in classes:
        file_name = f'./data/{lib}/{classe}_{dataset}.csv'
        if verbose:
            log.info(f'[{label_count}] Classe {classe}: {file_name}')
        if os.path.isfile(file_name):
            if ratio:
                data = load_by_chunks(file_name)
            else:
                data = pd.read_csv(file_name)
                data = remove_feature(data, 'img_name')
                data = remove_feature(data, 'id')

            log.info(f"Classe {classe}: {data.shape}")
            label = label_count * np.ones(len(data), dtype=np.int)

            X = merge_frames([X, data])
            y = np.concatenate((y, label))
        else:
            log.info("File not found for parameters: [{0}, {1}, {2}, {3}]".format(lib, dataset, classes, ratio))

        label_count += 1

    X, y = shuffle(X, y, random_state=random_state)
    return X, y.astype('int64')


def load(folder='casos', dataset='distances_all_px_eu', label_name='labels'):
    if folder == '':
        path = "./{0}.csv".format(dataset)
    else:
        path = "./data/{0}/{1}.csv".format(folder, dataset)

    if os.path.isfile(path):
        data = pd.read_csv(path)

        labels = data[label_name].values
        data = remove_feature(data, [label_name])

        return data, labels
    else:
        raise IOError("File not found")

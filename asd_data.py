from config import logger
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle

import initContext as context
context.loadModules()
log = logger.getLogger(__file__)
random_state = 10000


def load_data(lib='dlibHOG', dataset='distances_all_px_eu', classes=None, verbose=True):
    if classes is None or len(classes) == 1:
        raise IOError('It is not possible to load a dataset with {0} argument. Please insert two or more classes names'.format(classes))

    if verbose:
        log.info("Loading data from csv file")

    X = pd.DataFrame()
    y = np.array([])

    label_count = 0
    for classe in classes:
        file_name = './data/{0}/{1}_{2}.csv'.format(lib, classe, dataset)
        if verbose:
            log.info('[{0}] Classe {1}: {2}'.format(label_count, classe, file_name))
        if os.path.isfile(file_name):
            if lib == 'ratio':
                chuncksize, chunklist = 10, []
                for chunk in pd.read_csv(file_name, chunksize=chuncksize, dtype=np.float64):
                    chunklist.append(chunk)
                data = pd.concat(chunklist)
            else:
                data = pd.read_csv(file_name)
                data = data.drop(['img_name', 'id'], axis=1)

            log.info("Classe {0}: {1}".format(classe, data.shape))
            label = label_count * np.ones(len(data), dtype=np.int)

            X = pd.concat([X, data])
            y = np.concatenate((y, label))
        else:
            log.info("File not found for parameters: [{0}, {1}, {2}]".format(lib, dataset, classes))

        label_count += 1

    X, y = shuffle(X, y, random_state=random_state)
    return X, y.astype('int64')


def load_all(folder='casos', dataset='distances_all_px_eu', label_name='labels'):
    if folder == '':
        path = "./{0}.csv".format(dataset)
    else:
        path = "./data/{0}/{1}.csv".format(folder, dataset)

    if os.path.isfile(path):
        data = pd.read_csv(path)

        labels = data[label_name].values
        data = data.drop(label_name, axis=1)

        return data, labels
    else:
        raise IOError("File not found")

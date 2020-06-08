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
    for chunk in pd.read_csv(file_name, chunksize=chuncksize):
        chunk_list.append(chunk)
    return merge_frames(chunk_list)


def load_data(d_type="euclidian", unit="px", dataset="all", m="", normalization=False, labels=False, verbose=True):

    if verbose:
        log.info("Loading data from csv file")
    filename = "/distances"

    if normalization:
        filename = filename + "_ratio"

    if dataset == 'all' or dataset == 'farkas':
        filename = filename + "_" + dataset

    if unit == 'px' or unit == 'cm':
        filename = filename + "_" + unit

    if m == '1000':
        filename = filename + "_" + m

    if d_type == 'euclidian' or d_type == 'manhattan':
        filename = filename + "_eu"

    casos_file = cdir.CASES_DIR + filename + ".csv"
    if verbose:
        log.info("Casos file: " + str(casos_file))

    controles_file = cdir.CONTROL_DIR_1 + filename + ".csv"
    if verbose:
        log.info("Controles file: " + str(controles_file))

    if os.path.isfile(casos_file) and os.path.isfile(controles_file):
        if normalization:
            casos = load_by_chunks(casos_file)
            controles = load_by_chunks(controles_file)
        else:
            casos = pd.read_csv(casos_file, delimiter=',')
            controles = pd.read_csv(controles_file, delimiter=',')

        casos_label = np.ones(len(casos), dtype=np.int)
        controles_label = np.zeros(len(controles), dtype=np.int)

        if labels:
            casos['class'] = casos_label
            controles['class'] = controles_label

        # merge dataframes
        frames = [casos, controles]
        X = merge_frames(frames)

        # remove image paths
        if not normalization:
            X = remove_feature(X, 'img_name')
            X = remove_feature(X, 'id')

        target = np.concatenate((casos_label, controles_label))

        if not __checkDimension(X, target):
            raise ValueError("X and Y dimensions are not the same: {0} - {1}".format(X.shape, target.shape))

        if labels:
            return shuffle(X, random_state=random_state)
        else:
            X, target = shuffle(X, target, random_state=random_state)
            return X, target
    else:
        raise IOError("File not found for parameters: [{0}, {1}, {2}, {3}]".format(dataset, unit, m, d_type))


def load_wine():
    dataset = pd.read_csv('./data/wine.csv', delimiter=',')
    target = dataset['class_label'].values

    X = remove_feature(dataset, 'class_label')

    return X, target


def load_glass():
    dataset = pd.read_csv('./data/glass.csv', delimiter=',')
    target = dataset['class_label'].values

    X = remove_feature(dataset, 'class_label')

    return X, target


def load(dataset='distances_all_px_eu', folder='casos', feature_to_remove=None, label=1):
    if feature_to_remove is None:
        feature_to_remove = ['img_name', 'id']
    if folder == '' and label == '':
        path = "./{0}.csv".format(dataset)
    else:
        path = "./data/{0}/{1}.csv".format(folder, dataset)

    data = pd.read_csv(path)

    if feature_to_remove is not None:
        data = remove_feature(data, feature_to_remove)
    labels = np.full(len(data), label)

    return data, labels

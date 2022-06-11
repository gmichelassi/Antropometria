import csv
import math
import numpy as np
import os
import pandas as pd

from antropometria.exceptions import NonBinaryDatasetError
from scipy import stats
from typing import List


def apply_pearson_feature_selection(samples: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    if threshold >= 1.0 or threshold <= 0.0:
        raise ValueError(f'Expected values 0.0 < x < 1.0, received x={threshold}')

    n_features = samples.shape[1]
    features_to_delete = np.zeros(n_features, dtype=bool)

    for i in range(0, n_features):
        if not features_to_delete[i]:
            feature_i = samples.iloc[:, i].to_numpy()

            for j in range(i+1, n_features):
                if not features_to_delete[j]:
                    feature_j = samples.iloc[:, j].to_numpy()
                    pearson, pvalue = stats.pearsonr(feature_i, feature_j)
                    if abs(pearson) >= threshold:
                        features_to_delete[j] = True

    return samples[samples.columns[~features_to_delete]]


def combine_columns_names(n_columns: int, columns_names: pd.Index, mode: str = 'default') -> list:
    names = []
    if mode == 'default':
        for i in range(0, n_columns):
            for j in range(i+1, n_columns):
                names.append(f"{i}/{j}")
    elif mode == 'complete':
        for i in range(0, n_columns):
            for j in range(i+1, n_columns):
                names.append(f"{columns_names[i]}/{columns_names[j]}")

    return names


def build_ratio_dataset(dataset: pd.DataFrame, name: str) -> None:
    n_linhas, n_columns = dataset.shape
    linha_dataset_final = []

    columns = combine_columns_names(n_columns=n_columns, columns_names=dataset.columns, mode='complete')
    folder_name = './antropometria/data/ratio'

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    with open(f'{folder_name}/{name}.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)

        for linha in range(0, n_linhas):
            for coluna_i in range(0, n_columns):
                valor_i = dataset.iloc[linha, coluna_i]

                for coluna_j in range(coluna_i + 1, n_columns):
                    valor_j = dataset.iloc[linha, coluna_j]
                    if valor_i == 0.0 or valor_j == 0.0 or valor_i == np.nan or valor_j == np.nan or valor_i == np.Inf \
                            or valor_j == valor_j == np.Inf:
                        ratio_dist = 0.0
                    else:
                        if valor_i >= valor_j:
                            ratio_dist = valor_i / valor_j
                        else:
                            ratio_dist = valor_j / valor_i

                    linha_dataset_final.append(np.float64(ratio_dist))

            writer.writerow(linha_dataset_final)
            linha_dataset_final.clear()


def apply_min_max_normalization(df: pd.DataFrame) -> pd.DataFrame:
    df_final = []

    max_dist = math.ceil(np.amax(df.to_numpy()))
    min_dist = math.floor(np.amin(df.to_numpy()))

    for feature, data in df.iteritems():
        columns = []
        for i in data.values:
            xi = (i - min_dist)/(max_dist - min_dist)
            columns.append(xi)
        df_final.append(columns)

    return pd.DataFrame(df_final, dtype=float).T


def get_difference_of_classes(classes_count: List[int]) -> int:
    if len(classes_count) != 2:
        raise NonBinaryDatasetError(len(classes_count))

    return abs(classes_count[0] - classes_count[1])

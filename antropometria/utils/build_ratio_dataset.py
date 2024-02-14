import csv
import numpy as np
import os
import pandas as pd

from antropometria.utils.combine_columns_names import combine_columns_names


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

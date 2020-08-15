import asd_data as asd

# Utils
from utils.utils import apply_pearson_feature_selection, build_ratio_dataset
from scipy import stats
import os
import time
import numpy as np
import pandas as pd
import initContext as context

from multiprocessing import Process
from config import logger

context.loadModules()
log = logger.getLogger(__file__)


def splitDataFrame():
    #  https://stackoverflow.com/questions/48476629/how-to-split-dataframe-vertically-having-n-columns-in-each-resulting-df#comment83947161_48476787
    #  https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.split.html
    #  https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html

    X, y = asd.load_data(lib='ratio', dataset='distances_all_px_eu', classes=['casos', 'controles'], verbose=True)

    N = 33780
    log.info("Splitting dataset")

    log.info("X.shape %s, y.shape %s", str(X.shape), str(y.shape))

    dfs = np.split(X, np.arange(N, len(X.columns), N), axis=1)
    indexes = list(range(0, len(dfs)))

    for subDataFrame, index in zip(dfs, indexes):
        print(subDataFrame.shape)
        subDataFrame.to_csv(path_or_buf=f"./data/subDataSets/distances_eu_{index}.csv", index=False)


def runPearsonCorrelation(starting_file=0, ending_file=64, filtro=0.99, merge=False, contador=0):
    where_to_start = 0

    for i in range(starting_file, ending_file):
        log.info("Processing file {0} out of {1}".format(i, ending_file - 1))

        if merge:
            where_to_start = mergeDataFrames(i, i+1)
            contador += 1

        current_split = pd.read_csv(filepath_or_buffer=f'./data/subDataSets/distances_eu_{i}.csv')

        log.info("Applying pearson correlation filter")
        if where_to_start == 0:
            samples = apply_pearson_feature_selection(current_split, filtro)
        else:
            samples = custom_pearson_feature_selection(current_split, filtro, where_to_start)

        log.info("Saving file!")
        pd.DataFrame(data=samples).to_csv(path_or_buf=f'./data/subDataSets/processed-distances_eu_{i}.csv')

        log.info("Removing unused files...")
        os.remove(f'./data/subDataSets/distances_eu_{i}.csv')

        log.info("Done...")


def mergeDataFrames(i=0, j=1, indice=0):
        file_name1 = f"distances_eu_{i}.csv"
        file_name2 = f"distances_eu_{j}.csv"

        df1 = pd.read_csv(filepath_or_buffer='./data/subDataSets/processed-{0}'.format(file_name1))
        df2 = pd.read_csv(filepath_or_buffer='./data/subDataSets/processed-{0}'.format(file_name2))

        frames = [df1, df2]

        final_df = pd.concat(frames, axis=1, ignore_index=True)
        print("file {0} - shape {1}".format(indice, final_df.shape))

        log.info("Saving file!")

        final_df.to_csv(path_or_buf=f"./data/subDataSets/distances_eu_{indice}.csv")

        log.info("Removing unused files...")
        os.remove('./data/subDataSets/processed-{0}'.format(file_name1))
        os.remove('./data/subDataSets/processed-{0}'.format(file_name2))

        log.info("Done...")
        return df1.shape[1]


def custom_pearson_feature_selection(samples, max_value=0.99, where_to_start=1):
    columns_name = samples.columns  # Salvamos os "nomes" de todas colunas para identifica-las
    features_to_delete = []
    n_features = samples.shape[1]

    for i in range(0, n_features):
        if i == where_to_start:
            break

        if columns_name[i] not in features_to_delete:
            feature_i = samples[columns_name[i]].to_numpy()
            for j in range(where_to_start, n_features):
                if columns_name[j] not in features_to_delete:
                    feature_j = samples[columns_name[j]].to_numpy()
                    pearson, pvalue = stats.pearsonr(feature_i, feature_j)
                    if pearson >= max_value:
                        features_to_delete.append(columns_name[j])

    return samples.drop(features_to_delete, axis=1).values


if __name__ == '__main__':
    X = pd.read_csv('./data/dlibHOG/casos_distances_all_px_eu.csv')
    X = X.drop('img_name', axis=1)
    X = X.drop('id', axis=1)
    build_ratio_dataset(X, 'casos')

    X = pd.read_csv('./data/dlibHOG/controles_distances_all_px_eu.csv')
    X = X.drop('img_name', axis=1)
    X = X.drop('id', axis=1)
    build_ratio_dataset(X, 'controles')

    splitDataFrame()

    proc1 = Process(target=runPearsonCorrelation, args=(0, 7, False, 0))
    proc2 = Process(target=runPearsonCorrelation, args=(8, 15, False, 0))
    proc3 = Process(target=runPearsonCorrelation, args=(16, 23, False, 0))
    proc4 = Process(target=runPearsonCorrelation, args=(24, 31, False, 0))
    proc5 = Process(target=runPearsonCorrelation, args=(32, 39, False, 0))
    proc6 = Process(target=runPearsonCorrelation, args=(40, 47, False, 0))
    proc7 = Process(target=runPearsonCorrelation, args=(48, 55, False, 0))
    proc8 = Process(target=runPearsonCorrelation, args=(56, 63, False, 0))

    proc1.start()
    proc2.start()
    proc3.start()
    proc4.start()
    proc5.start()
    proc6.start()
    proc7.start()
    proc8.start()

    proc1.join()
    proc2.join()
    proc3.join()
    proc4.join()
    proc5.join()
    proc6.join()
    proc7.join()
    proc8.join()

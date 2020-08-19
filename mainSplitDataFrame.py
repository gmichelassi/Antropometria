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

    N = 33785
    log.info("Splitting dataset")

    log.info("X.shape %s, y.shape %s", str(X.shape), str(y.shape))

    dfs = np.split(X, np.arange(N, len(X.columns), N), axis=1)
    indexes = list(range(0, len(dfs)))

    for subDataFrame, index in zip(dfs, indexes):
        print(subDataFrame.shape)
        subDataFrame.to_csv(path_or_buf='./data/subDataSets/distances_eu_{0}.csv'.format(index), index=False)


def runPearsonCorrelation(starting_file=0, ending_file=64, filtro=0.99, merge=False, contador=0, indice=0):
    where_to_start = 0

    for i in range(starting_file, ending_file):
        log.info("Processing file {0} out of {1}".format(i, ending_file - 1))

        if merge:
            where_to_start = mergeDataFrames(contador, contador+1, indice)
            contador += 2
            indice += 1

        current_split = pd.read_csv(filepath_or_buffer='./data/subDataSets/distances_eu_{0}.csv'.format(i))

        log.info("Applying pearson correlation filter")
        if where_to_start == 0:
            samples = apply_pearson_feature_selection(current_split, filtro)
        else:
            samples = custom_pearson_feature_selection(current_split, filtro, where_to_start)

        log.info("Saving file!")
        pd.DataFrame(data=samples).to_csv(path_or_buf='./data/subDataSets/processed-distances_eu_{0}.csv'.format(i))

        log.info("Removing unused files...")
        os.remove('./data/subDataSets/distances_eu_{0}.csv'.format(i))

        log.info("Done...")


def mergeDataFrames(file_i=0, file_j=1, indice=0):
    file_name1 = "distances_eu_{0}.csv".format(file_i)
    file_name2 = "distances_eu_{0}.csv".format(file_j)

    df1 = pd.read_csv(filepath_or_buffer='./data/subDataSets/processed-{0}'.format(file_name1))
    df2 = pd.read_csv(filepath_or_buffer='./data/subDataSets/processed-{0}'.format(file_name2))

    frames = [df1, df2]

    final_df = pd.concat(frames, axis=1, ignore_index=True)
    print("file {0} - shape {1}".format(indice, final_df.shape))

    log.info("Saving file!")

    final_df.to_csv(path_or_buf="./data/subDataSets/distances_eu_{0}.csv".format(indice))

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


def build_data():
    X = pd.read_csv('./data/dlibHOG/casos_distances_all_px_eu.csv')
    X = X.drop('img_name', axis=1)
    X = X.drop('id', axis=1)
    build_ratio_dataset(X, 'casos')
    X = pd.read_csv('./data/dlibHOG/controles_distances_all_px_eu.csv')
    X = X.drop('img_name', axis=1)
    X = X.drop('id', axis=1)
    build_ratio_dataset(X, 'controles')


def nivel7():
    processes = [
        Process(target=runPearsonCorrelation, args=(0, 8, 0.95, False, 0, 0)),
        Process(target=runPearsonCorrelation, args=(8, 16, 0.95, False, 0, 0)),
        Process(target=runPearsonCorrelation, args=(16, 24, 0.95, False, 0, 0)),
        Process(target=runPearsonCorrelation, args=(24, 32, 0.95, False, 0, 0)),
        Process(target=runPearsonCorrelation, args=(32, 40, 0.95, False, 0, 0)),
        Process(target=runPearsonCorrelation, args=(40, 48, 0.95, False, 0, 0)),
        Process(target=runPearsonCorrelation, args=(48, 56, 0.95, False, 0, 0)),
        Process(target=runPearsonCorrelation, args=(56, 64, 0.95, False, 0, 0))
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()


def nivel6():
    processes = [
        Process(target=runPearsonCorrelation, args=(0, 4, 0.95, True, 0, 0)),
        Process(target=runPearsonCorrelation, args=(4, 8, 0.95, True, 8, 4)),
        Process(target=runPearsonCorrelation, args=(8, 12, 0.95, True, 16, 8)),
        Process(target=runPearsonCorrelation, args=(12, 16, 0.95, True, 24, 12)),
        Process(target=runPearsonCorrelation, args=(16, 20, 0.95, True, 32, 16)),
        Process(target=runPearsonCorrelation, args=(20, 24, 0.95, True, 40, 20)),
        Process(target=runPearsonCorrelation, args=(24, 28, 0.95, True, 48, 24)),
        Process(target=runPearsonCorrelation, args=(28, 32, 0.95, True, 56, 28))
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()


def nivel5():
    pass


if __name__ == '__main__':
    # build_data()

    start_time = time.time()
    log.info("--- Total splitting time: %s minutes ---" % ((time.time() - start_time) / 60))
    splitDataFrame()

    log.info("Processing nivel 7")
    start_time = time.time()
    nivel7()
    log.info("--- Total processing 7 time: %s minutes ---" % ((time.time() - start_time) / 60))

    log.info("Processing nivel 6")
    start_time = time.time()
    nivel6()
    log.info("--- Total processing 6 time: %s minutes ---" % ((time.time() - start_time) / 60))

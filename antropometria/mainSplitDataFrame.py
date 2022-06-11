import numpy as np
import os
import pandas as pd
import sys
import time

from config import logger
from multiprocessing import Process
from scipy import stats
from antropometria.utils.load_data import LoadData
from utils.dataset.manipulation import apply_pearson_feature_selection, build_ratio_dataset

log = logger.get_logger(__file__)

FOLDER = 'ratio'
DATASET_NAME = 'distances_all_px_eu'
CLASSES = ['casos', 'controles']


def split_data_frame(num_of_columns_per_split=33785):
    x, y = LoadData(FOLDER, DATASET_NAME, CLASSES).load()

    log.info("Splitting dataset")

    dfs = np.split(X, np.arange(num_of_columns_per_split, len(X.columns), num_of_columns_per_split), axis=1)
    indexes = list(range(0, len(dfs)))

    for subDataFrame, index in zip(dfs, indexes):
        if not os.path.exists(f'data/{FOLDER}/subDataSet'):
            os.mkdir(f'data/{FOLDER}/subDataSet')

        subDataFrame.to_csv(path_or_buf=f'data/{FOLDER}/subDataSet/{DATASET_NAME}_{index}.csv', index=False)

    pd.DataFrame(y).to_csv(f'data/{FOLDER}/subDataSet/label_{DATASET_NAME}.csv', index=False)
    log.info("Splitting complete")


def run_pearson_correlation(starting_file=0, ending_file=64, filtro=0.99, where_to_start=None):

    for indice in range(starting_file, ending_file):
        log.info("Processing file {0} out of {1}".format(indice, ending_file - 1))

        current_split = pd.read_csv(filepath_or_buffer=f'data/{FOLDER}/subDataSet/{DATASET_NAME}_{indice}.csv')

        if where_to_start is None:
            samples = apply_pearson_feature_selection(current_split, filtro)
        else:
            samples = custom_pearson_feature_selection(current_split, filtro, where_to_start[indice])

        pd.DataFrame(data=samples).to_csv(
            path_or_buf=f'data/{FOLDER}/subDataSet/processed_{DATASET_NAME}_{indice}.csv', index=False)
        os.remove(f'data/{FOLDER}/subDataSet/{DATASET_NAME}_{indice}.csv')


def merge_data_frames(indice_i, indice_j, new_indice):
    file_name1 = f"{DATASET_NAME}_{indice_i}.csv"
    file_name2 = f"{DATASET_NAME}_{indice_j}.csv"

    df1 = pd.read_csv(filepath_or_buffer=f'data/{FOLDER}/subDataSet/processed_{file_name1}')
    df2 = pd.read_csv(filepath_or_buffer=f'data/{FOLDER}/subDataSet/processed_{file_name2}')

    where_to_start = df1.shape[1]
    frames = [df1, df2]
    final_df = pd.concat(frames, axis=1)

    final_df.to_csv(path_or_buf=f"data/{FOLDER}/subDataSet/{DATASET_NAME}_{new_indice}.csv", index=False)

    os.remove(f'data/{FOLDER}/subDataSet/processed_{file_name1}')
    os.remove(f'data/{FOLDER}/subDataSet/processed_{file_name2}')

    return where_to_start


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

    return samples.drop(features_to_delete, axis=1)


def build_data():
    for class_name in CLASSES:
        file_name = f'data/{FOLDER}/{class_name}_{DATASET_NAME}.csv'

        df = pd.read_csv(file_name)
        if 'dlibHOG' in FOLDER:
            df = df.drop('img_name', axis=1)
            df = df.drop('id', axis=1)
        build_ratio_dataset(df, class_name)


def nivel7():
    processes = [
        Process(target=run_pearson_correlation, args=(0, 8, 0.95, None)),
        Process(target=run_pearson_correlation, args=(8, 16, 0.95, None)),
        Process(target=run_pearson_correlation, args=(16, 24, 0.95, None)),
        Process(target=run_pearson_correlation, args=(24, 32, 0.95, None)),
        Process(target=run_pearson_correlation, args=(32, 40, 0.95, None)),
        Process(target=run_pearson_correlation, args=(40, 48, 0.95, None)),
        Process(target=run_pearson_correlation, args=(48, 56, 0.95, None)),
        Process(target=run_pearson_correlation, args=(56, 64, 0.95, None))
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()


def nivel6():
    where_to_start, new_indice = [], 0
    for i in range(0, 64, 2):
        where_to_start.append(merge_data_frames(i, i + 1, new_indice))
        new_indice += 1

    processes = [
        Process(target=run_pearson_correlation, args=(0, 4, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(4, 8, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(8, 12, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(12, 16, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(16, 20, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(20, 24, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(24, 28, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(28, 32, 0.95, where_to_start))
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()


def nivel5():
    where_to_start, new_indice = [], 0
    for i in range(0, 32, 2):
        where_to_start.append(merge_data_frames(i, i + 1, new_indice))
        new_indice += 1
    print(where_to_start)
    processes = [
        Process(target=run_pearson_correlation, args=(0, 2, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(2, 4, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(4, 6, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(6, 8, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(8, 10, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(10, 12, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(12, 14, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(14, 16, 0.95, where_to_start))
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()


def nivel4():
    where_to_start, new_indice = [], 0
    for i in range(0, 16, 2):
        where_to_start.append(merge_data_frames(i, i + 1, new_indice))
        new_indice += 1
    print(where_to_start)

    processes = [
        Process(target=run_pearson_correlation, args=(0, 1, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(1, 2, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(2, 3, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(3, 4, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(4, 5, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(5, 6, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(6, 7, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(7, 8, 0.95, where_to_start))
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()


def nivel3():
    where_to_start, new_indice = [], 0
    for i in range(0, 8, 2):
        where_to_start.append(merge_data_frames(i, i + 1, new_indice))
        new_indice += 1
    print(where_to_start)

    processes = [
        Process(target=run_pearson_correlation, args=(0, 1, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(1, 2, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(2, 3, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(3, 4, 0.95, where_to_start)),
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()


def nivel2():
    where_to_start, new_indice = [], 0
    for i in range(0, 4, 2):
        where_to_start.append(merge_data_frames(i, i + 1, new_indice))
        new_indice += 1
    print(where_to_start)

    processes = [
        Process(target=run_pearson_correlation, args=(0, 1, 0.95, where_to_start)),
        Process(target=run_pearson_correlation, args=(1, 2, 0.95, where_to_start)),
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()


def nivel1():
    where_to_start, new_indice = [], 0
    for i in range(0, 2, 2):
        where_to_start.append(merge_data_frames(i, i + 1, new_indice))
        new_indice += 1
    print(where_to_start)

    processes = [
        Process(target=run_pearson_correlation, args=(0, 1, 0.95, where_to_start)),
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    args = sys.argv

    if args[1] == 'buildata':
        build_data()
    elif args[1] == 'split':
        start_time = time.time()
        split_data_frame()
        log.info("--- Total splitting time: %s minutes ---" % ((time.time() - start_time) / 60))
    elif args[1] == 'nivel7':
        log.info("Processing nivel 7")
        start_time = time.time()
        nivel7()
        log.info("--- Total processing 7 time: %s minutes ---" % ((time.time() - start_time) / 60))
    elif args[1] == 'nivel6':
        log.info("Processing nivel 6")
        start_time = time.time()
        nivel6()
        log.info("--- Total processing 6 time: %s minutes ---" % ((time.time() - start_time) / 60))
    elif args[1] == 'nivel5':
        log.info("Processing nivel 5")
        start_time = time.time()
        nivel5()
        log.info("--- Total processing 5 time: %s minutes ---" % ((time.time() - start_time) / 60))
    elif args[1] == 'nivel4':
        log.info("Processing nivel 4")
        start_time = time.time()
        nivel4()
        log.info("--- Total processing 4 time: %s minutes ---" % ((time.time() - start_time) / 60))
    elif args[1] == 'nivel3':
        log.info("Processing nivel 3")
        start_time = time.time()
        nivel3()
        log.info("--- Total processing 3 time: %s minutes ---" % ((time.time() - start_time) / 60))
    elif args[1] == 'nivel2':
        log.info("Processing nivel 2")
        start_time = time.time()
        nivel2()
        log.info("--- Total processing 2 time: %s minutes ---" % ((time.time() - start_time) / 60))
    elif args[1] == 'nivel1':
        log.info("Processing nivel 1")
        start_time = time.time()
        nivel1()
        log.info("--- Total processing 1 time: %s minutes ---" % ((time.time() - start_time) / 60))
    elif args[1] == 'test':
        log.info('Parâmetros estão sendo recebidos com sucesso')
        X = pd.read_csv('data/ratio/subDataSet/processed_distances_all_px_eu_0.csv')
        print(X.head())
    else:
        log.error('Parâmetro não encontrado')

# DataSets
import asd_data as asd

# Utils
from classifiers.utils import apply_pearson_feature_selection
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
    all_samples = {}

    X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='all', labels=False)
    all_samples['euclidian_px_all'] = (X, y)

    # X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='all', normalization='ratio', labels=False)
    # all_samples['euclidian_ratio_px_all'] = (X, y)

    for k in all_samples.keys():
        N = 759
        log.info("Spliting %s dataset", k)
        samples, labels = all_samples[k]

        log.info("X.shape %s, y.shape %s", str(samples.shape), str(labels.shape))

        dfs = np.split(samples, np.arange(N, len(samples.columns), N), axis=1)
        indexes = list(range(0, len(dfs)))

        for subDataFrame, index in zip(dfs, indexes):
            print(subDataFrame.shape)
            subDataFrame.to_csv(path_or_buf="./data/subDataSets/{0}-{1}.csv".format(k, index),
                                index=False)


def runPearsonCorrelation(dataset_name, starting_file=0, ending_file=64, filtro=0.99, where_to_start=0):
    # range é um intervalo [inicio, fim)
    for i in range(starting_file, ending_file):
        log.info("Processing file {0} out of {1}".format(i, ending_file - 1))  # -1 pois o intervalo do for eh aberto

        file_name = "{0}-{1}.csv".format(dataset_name, i)
        current_split = pd.read_csv(filepath_or_buffer='./data/subDataSets/{0}'.format(file_name))

        log.info("Applying pearson correlation filter")
        if where_to_start == 0:
            samples = apply_pearson_feature_selection(current_split, filtro)
        else:
            print("entrou no código certo :)", flush=True)
            samples = custom_pearson_feature_selection(current_split, filtro, where_to_start)

        current_split = pd.DataFrame(data=samples)

        log.info("Saving file!")
        current_split.to_csv(path_or_buf='./data/subDataSets/processed-{0}'.format(file_name))

        log.info("Removing unused files...")
        os.remove('./data/subDataSets/{0}'.format(file_name))

        log.info("Done...")


def mergeDataFrames(dataset_name, starting_file=0, ending_file=64):
    log.info("Merging already processed datasets pairwise...")
    contador = 0
    for i in range(starting_file, ending_file, 2):
        file_name1 = "{0}-{1}.csv".format(dataset_name, i)
        file_name2 = "{0}-{1}.csv".format(dataset_name, i + 1)

        df1 = pd.read_csv(filepath_or_buffer='./data/subDataSets/processed-{0}'.format(file_name1))
        df2 = pd.read_csv(filepath_or_buffer='./data/subDataSets/processed-{0}'.format(file_name2))

        frames = [df1, df2]

        final_df = pd.concat(frames, axis=1, ignore_index=True)
        print("file {0} - shape {1}".format(contador, final_df.shape))

        log.info("Saving file!")

        final_df.to_csv(path_or_buf="./data/subDataSets/{0}-{1}.csv".format(dataset_name, contador))

        log.info("Removing unused files...")
        os.remove('./data/subDataSets/processed-{0}'.format(file_name1))
        os.remove('./data/subDataSets/processed-{0}'.format(file_name2))

        log.info("Done...")
        contador += 1


def checkShapes(dataset_name, starting_file=0, ending_file=64):
    for i in range(starting_file, ending_file):
        file_name = "{0}-{1}.csv".format(dataset_name, i)
        df = pd.read_csv(filepath_or_buffer='./data/subDataSets/processed-{0}'.format(file_name))
        print("File {0} - Shape: {1} - Redução de {2:.2f}%".format(i, df.shape,
                                                                   100 - (round(df.shape[1] / 41000, 2) * 100)))


# mesmo método da correlação de pearson, porém com uma alteração para onde o segundo laço deve começar
# fazemos isso para evitar comparar features em excesso
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
    start_time = time.time()

    multipleProcessing = True

    if multipleProcessing:
        proc1 = Process(target=runPearsonCorrelation, args=("euclidian_ratio_px_all", 0, 1, 0.99, 65070))
        proc2 = Process(target=runPearsonCorrelation, args=("euclidian_ratio_px_all", 1, 2, 0.99, 70695))
        proc3 = Process(target=runPearsonCorrelation, args=("euclidian_ratio_px_all", 2, 3, 0.99, 66468))
        proc4 = Process(target=runPearsonCorrelation, args=("euclidian_ratio_px_all", 3, 4, 0.99, 69539))
        proc5 = Process(target=runPearsonCorrelation, args=("euclidian_ratio_px_all", 4, 5, 0.99, 73532))
        proc6 = Process(target=runPearsonCorrelation, args=("euclidian_ratio_px_all", 5, 6, 0.99, 70400))
        proc7 = Process(target=runPearsonCorrelation, args=("euclidian_ratio_px_all", 6, 7, 0.99, 68191))
        proc8 = Process(target=runPearsonCorrelation, args=("euclidian_ratio_px_all", 7, 8, 0.99, 74533))

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
    else:
        # checkShapes("euclidian_ratio_px_all", starting_file=0, ending_file=16)
        mergeDataFrames("euclidian_ratio_px_all")

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

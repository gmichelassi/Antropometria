import asd_data as asd

# Utils
from utils.utils import apply_pearson_feature_selection
from utils.utils import build_ratio_dataset
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

    # samples, labels = asd.load_data(lib='ratio', dataset='distances_all_px_eu', ratio=True, labels=False, verbose=True)
    samples, labels = pd.read_csv('./data/teste.csv'), np.array([1])
    N = 1

    log.info("X.shape %s, y.shape %s", str(samples.shape), str(labels.shape))

    dfs = np.split(samples, np.arange(N, len(samples.columns), N), axis=1)
    indexes = list(range(0, len(dfs)))

    for subDataFrame, index in zip(dfs, indexes):
        print(subDataFrame.shape)
        subDataFrame.to_csv(path_or_buf=f"./data/tmp/distances_eu_{index}.csv", index=False)


def runPearsonCorrelation(i=0, filtro=0.98, where_to_start=0):
    file_name = f"./data/tmp/distances_eu_-{i}.csv"

    if not os.path.isfile(file_name):
        return

    current_split = pd.read_csv(filepath_or_buffer=file_name)

    log.info("Applying pearson correlation filter")
    if where_to_start == 0:
        samples = apply_pearson_feature_selection(current_split, filtro)
    else:
        samples = custom_pearson_feature_selection(current_split, filtro, where_to_start)

    log.info("Saving file!")
    pd.DataFrame(data=samples).to_csv(path_or_buf=file_name)

    log.info("Done...")


def mergeDataFrames(index):
    file_name1 = f"./data/tmp/distances_eu_{index}.csv"
    file_name2 = ""

    if not os.path.isfile(file_name1):
        return -1

    for i in range(index+1, countFiles()):
        file_name2 = f"./data/tmp/distances_eu_{i}.csv"
        if os.path.isfile(file_name2):
            break

    if not os.path.isfile(file_name2):
        return -1

    print(f"encontrou os dois aquivos {file_name1} e {file_name2}")

    df1 = pd.read_csv(filepath_or_buffer=file_name1)
    df2 = pd.read_csv(filepath_or_buffer=file_name2)

    N = df1.shape[1]

    frames = [df1, df2]

    final_df = pd.concat(frames, axis=1, ignore_index=True)

    log.info("Saving file!")

    final_df.to_csv(path_or_buf=file_name1)

    log.info("Removing unused files...")
    os.remove(file_name2)

    log.info("Done...")

    return N


# mesmo método da correlação de pearson, porém com uma alteração para onde o segundo laço deve começar
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


def countFiles():
    isfile = os.path.isfile
    join = os.path.join

    directory = './data/tmp/'
    return sum(1 for item in os.listdir(directory) if isfile(join(directory, item)))


def principal(n_files):
    if countFiles() > 1:
        for i in range(0, n_files):
            indice_inicial = mergeDataFrames(i)
            if indice_inicial == -1:
                print("deu problema, dps muda pro contrario")
                # runPearsonCorrelation(i=i, where_to_start=indice_inicial)
        # principal(n_files)
        print(f"viria a recursao: {n_files}")


if __name__ == '__main__':
    start_time = time.time()

    # garantir que a pasta tmp existe
    if not os.path.isdir('./data/tmp/'):
        os.mkdir('./data/tmp/')

    # splitar o dataset inicial
    splitDataFrame()

    # fazer a primeira iteração do filtro
    for i in range(countFiles()):
        pass
        # runPearsonCorrelation(i=i)

    # método recursivo pra automaticamente passar o filtro e mesclar os arquivos
    n_files = countFiles()
    print(n_files)
    principal(n_files)
    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
from sklearn.decomposition import PCA
from scipy import stats


def mean_scores(scores):
    mean_score = {}
    for score_key, score_value in scores.items():
        mean_score[score_key] = np.mean(score_value, axis=0)
    return mean_score


def sample_std(scores):
    mean = np.mean(scores)
    std = 0
    for i in scores:
        std += (i - mean) ** 2
    std = std/(len(scores) - 1)
    return math.sqrt(std)


def apply_pearson_feature_selection(samples, max_value=0.99):
    n_features = samples.shape[1]
    features_to_delete = np.zeros(n_features, dtype=bool)

    for i in range(0, n_features):
        if not features_to_delete[i]:
            feature_i = samples.iloc[:, i].to_numpy()

            for j in range(i+1, n_features):
                if not features_to_delete[j]:
                    feature_j = samples.iloc[:, j].to_numpy()
                    pearson, pvalue = stats.pearsonr(feature_i, feature_j)
                    if pearson >= max_value:
                        features_to_delete[j] = True

    return samples[samples.columns[~features_to_delete]]


def build_ratio_dataset(dataset, name):
    n_linhas, n_columns = dataset.shape  # obtemos o tamanho do dataset
    linha_dataset_final = []

    columns = combine_columns_names(n_columns=n_columns, columns_names=dataset.columns, mode='complete')

    with open(f'./data/ratio/{name}_distances_all_px_eu.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)

        for linha in range(0, n_linhas):
            for coluna_i in range(0, n_columns):
                valor_i = dataset.iloc[linha, coluna_i]

                for coluna_j in range(coluna_i + 1, n_columns):
                    valor_j = dataset.iloc[linha, coluna_j]
                    if valor_i == 0.0 or valor_j == 0.0 or valor_i == np.nan or valor_j == np.nan or valor_i == np.Inf \
                            or valor_j == valor_j == np.Inf:
                        print(valor_j)
                        print(valor_i)
                        ratio_dist = 0.0
                    else:
                        if valor_i >= valor_j:
                            ratio_dist = valor_i / valor_j
                        else:
                            ratio_dist = valor_j / valor_i

                    linha_dataset_final.append(np.float(ratio_dist))

            writer.writerow(linha_dataset_final)
            linha_dataset_final.clear()

            print("Linha {0} concluida".format(linha), flush=True)


def combine_columns_names(n_columns, columns_names, mode='default'):
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


def varianciaAcumuladaPCA(samples, labels, verbose=False):
    pca = PCA()

    pca.fit(samples, labels)
    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)

    for i, soma in enumerate(cum_var_exp):
        print("PC" + str(i+1) + " Cumulative variance: {0:.3f}%".format(cum_var_exp[i]*100))

    if verbose:
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, len(cum_var_exp) + 1), var_exp, align='center', label='individual variance explained',
                alpha=0.7)
        plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='cumulative variance explained',
                 color='red')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.xticks(np.arange(1, len(var_exp) + 1, 1))
        plt.legend(loc='center right')
        plt.savefig("./output/pca-explained-variance.png")


def normalizacao_min_max(df):
    df_final = []

    max_dist = math.ceil(df.max(axis=1).max())
    min_dist = 0

    for (feature, data) in df.iteritems():
        columns = []
        for i in data.values:
            xi = (i - min_dist)/(max_dist - min_dist)
            columns.append(xi)
        df_final.append(columns)

    return pd.DataFrame(df_final).T

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


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
def apply_pearson_feature_selection(samples, max_value=0.99):
    from datetime import datetime

    columns_name = samples.columns  # Salvamos os "nomes" de todas colunas para identifica-las
    features_to_delete = []
    n_features = samples.shape[1]

    # features = []  # eixo X
    # times = []  # eixo Y

    for i in range(0, n_features):  # a função range indica um intervalo [0, n_features)
        if columns_name[i] not in features_to_delete:  # Se a coluna i não está na lista de colunas a serem deletadas

            # features.append(i)  # adicionando feature no eixo x

            feature_i = samples[columns_name[i]].to_numpy()  # Pegamos o vetor relativo a coluna i
            # inicio = datetime.now()
            # inicio_format = inicio.strftime('%d/%m/%Y %H:%M:%S')
            # print(str(inicio_format) + " - Comparando feature " + str(i), flush=True)

            # how_many_features = 0
            for j in range(i+1, n_features):
                if columns_name[j] not in features_to_delete:  # Se a coluna j não está na lista de colunas a serem deletadas
                    feature_j = samples[columns_name[j]].to_numpy()  # Pegamos o vetor relativo a coluna j
                    pearson, pvalue = stats.pearsonr(feature_i, feature_j)  # Realizamos o calculo da correlação
                    if pearson >= max_value:  # Se a correlação for maior do que o valor máximo, incluimos na lista de features a serem deletadas
                        features_to_delete.append(columns_name[j]) # a operação de inclusao na lista é O(1), então não afeta o desempenho do codigo
                        # how_many_features += 1
            # fim = datetime.now()
            # fim_format = fim.strftime('%d/%m/%Y %H:%M:%S')
            # print(str(fim_format) + " - Features deletadas para a feature " + str(i) + ": " + str(how_many_features), flush=True)
            # duration = (fim - inicio).total_seconds()

            # times.append(duration)

    # plt.plot(features, times)
    # plt.xlabel("Features (index)")
    # plt.ylabel("Processing time duration (s)")
    # plt.title("Feature x Time to process")
    # plt.savefig("./output/featurexduration-pearson.png")
    # pd.DataFrame(features_to_delete).T.to_csv(path_or_buf='./features_to_delete.csv', index=False)  # para salvar as features que precisam ser deletadas
    return samples.drop(features_to_delete, axis=1)  # Deletamos todas as features selecionadas e retornamos o DataFrame


def build_ratio_dataset(dataset, name):
    n_linhas, n_columns = dataset.shape  # obtemos o tamanho do dataset
    linha_dataset_final = []  # lista auxiliar que conterá as linhas do dataset final
    columns = combine_columns_names(n_columns=n_columns)

    with open(f'./data/ratio/{name}_distances_all_px_eu.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)

        for linha in range(0, n_linhas):
            for coluna_i in range(0, n_columns):
                valor_i = dataset.iloc[linha, coluna_i]

                for coluna_j in range(coluna_i + 1, n_columns):
                    valor_j = dataset.iloc[linha, coluna_j]
                    if valor_i == 0.0 or valor_j == 0.0 or valor_i == np.nan or valor_j == np.nan or valor_i == np.Inf or valor_j == valor_j == np.Inf:
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


def combine_columns_names(n_columns):
    names = []
    for i in range(0, n_columns):
        for j in range(i+1, n_columns):
            names.append("f{0}/f{1}".format(i, j))

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
        plt.bar(range(1, len(cum_var_exp) + 1), var_exp, align='center', label='individual variance explained', alpha=0.7)
        plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='cumulative variance explained', color='red')
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

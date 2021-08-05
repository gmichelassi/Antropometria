import os
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from antropometria.utils.dataset.load import LoadData
from sklearn.decomposition import PCA


def run_pca_analysis():
    x, y = LoadData('dlibHOG', 'distances_all_px_eu', ['casos', 'controles']).load()
    pca = PCA()
    pca.fit(x, y)

    components = pd.DataFrame(pca.components_, columns=x.columns).abs()
    best_components = components.max()
    sorted_components = best_components.sort_values(ascending=False)
    sorted_components.head(15).to_csv('./antropometria/output/pca_best_features.csv')

    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)

    for i, soma in enumerate(cum_var_exp):
        print("PC" + str(i + 1) + " Cumulative variance: {0:.3f}%".format(cum_var_exp[i] * 100))

    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(cum_var_exp) + 1), var_exp, align='center', label='individual variance explained',
            alpha=0.7)
    plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='cumulative variance explained',
             color='red')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.xticks(np.arange(1, len(var_exp) + 1, 1))
    plt.legend(loc='center right')
    plt.savefig("antropometria/output/pca-explained-variance.png")


if __name__ == '__main__':
    start_time = time.time()
    run_pca_analysis()
    print("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))
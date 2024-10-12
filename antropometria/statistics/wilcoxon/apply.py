from itertools import combinations

import pandas as pd
from scipy.stats import wilcoxon

from antropometria.config import IMAGE_PROCESSING_LIBS

SORT_COLUMNS = ['classifier', 'red_dim', 'dataset_imbalance']
FILTER_COLUMNS = ['classifier', 'red_dim', 'dataset_imbalance', 'err_f1score']


def apply_wilcoxon():
    results = ['./antropometria/data/results_individual.csv', './antropometria/data/results_shared.csv']

    for result in results:
        print('\n\nPROCESSING: ', result)
        data = pd.read_csv(result)
        for first_lib, second_lib in combinations(IMAGE_PROCESSING_LIBS, 2):
            results_from_first_lib = data.query('img_lib == @first_lib').sort_values(by=SORT_COLUMNS)
            results_from_second_lib = data.query('img_lib == @second_lib').sort_values(by=SORT_COLUMNS)

            number_of_rows = min(results_from_first_lib.shape[0], results_from_second_lib.shape[0])

            filtered_results_from_first_lib = results_from_first_lib.head(number_of_rows)[FILTER_COLUMNS]
            filtered_results_from_second_lib = results_from_second_lib.head(number_of_rows)[FILTER_COLUMNS]

            f1_score_folds_from_first_lib = [
                float(i.replace(',', '.')) for i in filtered_results_from_first_lib['err_f1score'].values
            ]
            f1_score_folds_from_second_lib = [
                float(i.replace(',', '.')) for i in filtered_results_from_second_lib['err_f1score'].values
            ]

            pvalue = wilcoxon(f1_score_folds_from_first_lib, f1_score_folds_from_second_lib, correction=True)

            print(f'Wilcoxon for {first_lib} and {second_lib}: {pvalue}')

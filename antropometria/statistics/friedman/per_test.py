import csv
import pandas as pd

from .perform import perform
from antropometria.config.constants import CLASSIFIER_NAMES, REDUCTIONS_NAMES, PEARSONS, MIN_MAXS, SAMPLING_NAMES
from itertools import product


TESTS = product(CLASSIFIER_NAMES, REDUCTIONS_NAMES, PEARSONS, MIN_MAXS, SAMPLING_NAMES)


def friedman_per_test(data: pd.DataFrame):
    for classifier, red_dim, pearson, min_max, sampling in TESTS:
        try:
            query = f'classifier == "{classifier}" and ' \
                    f'red_dim == "{red_dim}" and ' \
                    f'pearson == "{pearson}" and ' \
                    f'min_max == "{min_max}" and ' \
                    f'sampling == "{sampling}"'

            statistic, pvalue = perform(data=data, query=query, column='f1score_folds', expected_amount=5)

            print(f'For {classifier, red_dim, pearson, min_max, sampling} '
                  f'we have statistic={statistic} and pvalue={pvalue}')

        except ValueError:
            print(f'Could not perform Friedman test for '
                  f'{classifier, red_dim, pearson, min_max, sampling}')


def save(classifier, red_dim, pearson, min_max, sampling, statistic, pvalue):
    with open('./antropometria/output/friedman/analysis_per_test.csv', 'a') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                'Classifier', 'Dimensionality Reduction', 'Pearson Filter', 'Min Max', 'Sampling', 'Statistic', 'pvalue'
            ]
        )

        row = {
            'Classifier': classifier,
            'Dimensionality Reduction': red_dim,
            'Pearson Filter': pearson,
            'Min Max': min_max,
            'Sampling': sampling,
            'Statistic': statistic,
            'pvalue': pvalue
        }
        writer.writerow(row)

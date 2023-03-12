import pandas as pd

from .transform import transform
from antropometria.config.constants import CLASSIFIER_NAMES
from scipy.stats import friedmanchisquare
from statsmodels.sandbox.stats.multicomp import multipletests


def apply_friedman_for_all(data: pd.DataFrame):
    transformed_data = transform(data)

    statistic, pvalue = friedmanchisquare(*transformed_data.values)

    print(f'For all tests we have statistic={statistic} and pvalue={pvalue}')


def apply_friedman_for_per_test(data: pd.DataFrame):
    pvalues = []
    for classifier in CLASSIFIER_NAMES:
        transformed_data = transform(data, classifiers=[classifier])
        _, pvalue = friedmanchisquare(*transformed_data.values)
        pvalues.append(pvalue)

    holm_conclusions, bonferroni_adjusted_pvalues, _, _ = multipletests(pvalues, alpha=.05, method='bonferroni')
    holm_conclusions, holm_adjusted_pvalues, _, _ = multipletests(pvalues, alpha=.05, method='holm')

    for classifier, pvalue, bonferroni_adjusted_pvalue, holm_adjusted_pvalue in zip(CLASSIFIER_NAMES, pvalues, bonferroni_adjusted_pvalues, holm_adjusted_pvalues):
        print(f'For {classifier} we have pvalue={pvalue}, bonferroni_adjusted_pvalue={bonferroni_adjusted_pvalue} and holm_adjusted_pvalue={holm_adjusted_pvalue}')


apply_friedman = {
    'all': apply_friedman_for_all,
    'per_test': apply_friedman_for_per_test
}


def friedman(mode = 'all'):

    results = ['./antropometria/data/results_individual.csv', './antropometria/data/results_shared.csv']

    for result in results:
        print('Processing: ', result)
        data = pd.read_csv(result)
        apply_friedman[mode](data)


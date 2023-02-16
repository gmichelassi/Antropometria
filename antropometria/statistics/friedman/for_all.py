import csv
import pandas as pd

from .perform import perform


def friedman_for_all(data: pd.DataFrame):
    try:
        statistic, pvalue = perform(data=data, query='', column='f1score_folds')

        print(f'For all tests we have statistic={statistic} and pvalue={pvalue}')

        save(statistic, pvalue)
    except ValueError:
        print(f'Could not perform Friedman test for all tests')


def save(statistic, pvalue):
    with open('./antropometria/output/friedman/analysis_for_all_tests.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Teste', 'Statistic', 'pvalue'])

        row = {
            'Teste': 'Todos Testes',
            'Statistic': statistic,
            'pvalue': pvalue
        }

        writer.writerow(row)

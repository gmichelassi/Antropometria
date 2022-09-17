import csv
import pandas as pd

from .perform import perform
from antropometria.config.constants import CLASSIFIER_NAMES


def friedman_per_classifier(data: pd.DataFrame):
    for classifier in CLASSIFIER_NAMES:
        try:
            query = f'classifier == "{classifier}"'

            statistic, pvalue = perform(data=data, query=query, column='f1score_folds')

            print(f'For {classifier} we have statistic={statistic} and pvalue={pvalue}')

            save(classifier, statistic, pvalue)
        except ValueError:
            print(f'Could not perform Friedman test for {classifier}.')


def save(classifier, statistic, pvalue):
    with open('./antropometria/output/friedman/analysis_per_classifier.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Classifier', 'Statistic', 'pvalue'])

        row = {
            'Classifier': classifier,
            'Statistic': statistic,
            'pvalue': pvalue
        }

        writer.writerow(row)

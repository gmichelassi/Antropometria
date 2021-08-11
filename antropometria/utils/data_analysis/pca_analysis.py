import os
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
import numpy as np
import pandas as pd
import time

from antropometria.utils.dataset.load import LoadData
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA

THRESHOLD = 0.1


class PCAAnalysis:
    def __init__(self, x: pd.DataFrame, y: np.ndarray):
        self.x = x
        self.y = y

        self.pca = PCA()
        self.pca.fit(x, y)
        self.components = pd.DataFrame(self.pca.components_, columns=x.columns).abs()

        self.n_features = x.shape[1]
        self.sqrt_features = int(np.sqrt(self.n_features))
        self.best_components = self.components[:self.sqrt_features]

    def compute_best_features(self):
        components_dict = {}
        for component_index, row in self.best_components.iterrows():
            best_features = [feature_name for feature_name, feature_value in row.iteritems() if
                             feature_value >= THRESHOLD]
            components_dict[f"component_{component_index}"] = best_features

        return components_dict

    @staticmethod
    def compute_t_student_test(component_dict: dict, data: list[pd.DataFrame] = None):
        if len(data) != 2:
            raise IOError(f'Expected len(data) = 2 but was {len(data)}')
        test = []
        for component in component_dict.keys():
            features = component_dict[component]
            if len(features) > 0:
                for feature in features:
                    first_class_feature = data[0][feature].values
                    first_class_feature_mean = first_class_feature.mean()
                    second_class_feature = data[1][feature].values
                    second_class_feature_mean = second_class_feature.mean()

                    statistic, pvalue = ttest_ind(first_class_feature, second_class_feature)
                    test.append([feature, component, first_class_feature_mean, second_class_feature_mean, pvalue])

        pd.DataFrame(test, columns=['Distancia', 'Componente', 'Media Casos', 'Media Controles', 'pvalor']).to_csv(
            'antropometria/output/test_t_student_pca.csv',
            index=False,
            columns=['Distancia', 'Componente', 'Media Casos', 'Media Controles', 'pvalor']
        )

    @staticmethod
    def save_dict(components_dict: dict, file: str):
        output_file = open(file, "w")
        json.dump(components_dict, output_file, indent=4)
        output_file.close()


def main():
    x, y = LoadData('dlibHOG', 'distances_all_px_eu', ['casos', 'controles']).load()
    pca_analysis = PCAAnalysis(x, y)

    components_dict = pca_analysis.compute_best_features()
    pca_analysis.save_dict(components_dict, "antropometria/output/pca_components.json")

    casos = pd.read_csv('antropometria/data/dlibHOG/casos_distances_all_px_eu.csv')
    controles = pd.read_csv('antropometria/data/dlibHOG/controles_distances_all_px_eu.csv')
    pca_analysis.compute_t_student_test(components_dict, [casos, controles])


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))
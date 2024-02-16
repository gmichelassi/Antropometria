import numpy as np
import pandas as pd
import pytest

from antropometria.data.dataset_analyzer import DatasetAnalyzer

dataset = pd.DataFrame({
    'feature_1': [1, 2, 3, 4, 5],
    'feature_2': [6, 7, 8, 9, 10],
    'feature_3': [11, 12, 13, 14, 15],
})

labels = np.array(['class_1', 'class_2', 'class_1', 'class_1', 'class_2'])


class TestDatasetAnalyzer:
    def test_feature_names(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.feature_names() == ['feature_1', 'feature_2', 'feature_3']

    def test_class_names(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.class_names() == ['class_1', 'class_2']

    def test_number_of_features(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.number_of_features() == 3

    def test_number_of_instances(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.number_of_instances() == 5

    def test_number_of_classes(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.number_of_classes() == 2

    def test_mean_of(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.mean_of('feature_1') == 3.0

    def test_median_of(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.median_of('feature_1') == 3.0

    def test_std_of(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.std_of('feature_1') == 1.5811388300841898

    def test_min_of(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.min_of('feature_1') == 1

    def test_max_of(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.max_of('feature_1') == 5

    def test_count_of(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.count_of('feature_1') == 5

    def test_class_distribution(self):
        analyzer = DatasetAnalyzer(dataset, labels)

        assert analyzer.class_distribution() == {'class_1': 3, 'class_2': 2}

    def test_raises_value_error_when_labels_do_not_correspond_to_dataset(self):
        mismatch_labels = np.array(['class_1', 'class_2', 'class_1', 'class_1'])

        with pytest.raises(ValueError):
            DatasetAnalyzer(dataset, mismatch_labels)

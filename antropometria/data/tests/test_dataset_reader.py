from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from antropometria.data.dataset_reader import DatasetReader
from antropometria.exceptions import MissingDatasetError

dataset = pd.DataFrame({
    'image_name': ['image_1', 'image_2', 'image_3', 'image_4', 'image_5'],
    'label': ['class_1', 'class_2', 'class_1', 'class_1', 'class_2'],
    'feature_1': [1, 2, 3, 4, 5],
    'feature_2': [6, 7, 8, 9, 10],
    'feature_3': [11, 12, 13, 14, 15],
})


@patch('pandas.read_csv', return_value=dataset)
class TestDatasetReader:
    @patch('os.path.isfile', return_value=True)
    def test_load_data_from_single_file_correctly(self, *_):
        folder = 'folder'
        dataset_name = 'dataset'

        x, y, count = DatasetReader(folder, dataset_name).read()

        print(y)

        assert x.columns.tolist() == dataset.columns.tolist()
        assert np.unique(y).tolist() == dataset['label'].unique().tolist()
        assert count == [3, 2]

    @patch('os.path.isfile', side_effect=[False, *[True] * 10])
    def test_load_data_from_multiple_files_correctly(self, *_):
        folder = 'folder'
        dataset_name = 'dataset'
        classes = ['class_1', 'class_2']

        x, y, count = DatasetReader(folder, dataset_name, classes).read()

        assert x.columns.tolist() == dataset.columns.tolist()
        assert np.unique(y).tolist() == dataset['label'].unique().tolist()
        assert count == [6, 4]

    @patch('os.path.isfile', side_effect=[False, *[True] * 10])
    def test_drop_columns_correctly(self, *_):
        folder = 'folder'
        dataset_name = 'dataset'
        classes = ['class_1', 'class_2']
        columns_to_drop = ['image_name', 'label']

        x, _, _ = DatasetReader(folder, dataset_name, classes, columns_to_drop).read()

        assert x.columns.tolist() == ['feature_1', 'feature_2', 'feature_3']

    def test_raises_missing_dataset_error(self, *_):
        folder = 'folder'
        dataset_name = 'dataset'

        with pytest.raises(MissingDatasetError):
            DatasetReader(folder, dataset_name).read()

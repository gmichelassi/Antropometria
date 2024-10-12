import numpy as np
import pandas as pd

from antropometria.main import main, run
from unittest.mock import patch, call, Mock

import pytest

dataset = pd.DataFrame({})
labels = np.array([])
classes_count = [25, 24]


class DatasetReaderMock:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def read():
        return dataset, labels, classes_count


@patch('antropometria.main.run')
class TestMain:
    def test_main_when_data_param_is_present(self, mocked_run):
        data = [
            ('folder_1', 'dataset', ['class_1', 'class_2']),
            ('folder_2', 'dataset', ['class_1', 'class_2'])
        ]
        main(data=data)

        mocked_run.assert_has_calls(
            calls=[
                call('folder_1', 'dataset', ['class_1', 'class_2']),
                call('folder_2', 'dataset', ['class_1', 'class_2'])
            ],
            any_order=False
        )

    def test_main_when_folder_dataset_name_and_classes_params_are_present(self, mocked_run):
        main(folder='folder', dataset_name='dataset', classes=['class_1', 'class_2'])

        mocked_run.assert_called_once_with('folder', 'dataset', ['class_1', 'class_2'])

    def test_main_raises_value_error_when_invalid_arguments_are_passed(self, mocked_run):
        with pytest.raises(ValueError):
            main('folder', 'dataset', ['class_1', 'class_2'])

            mocked_run.assert_not_called()

    @patch('antropometria.main.cleanup_processed_data')
    def test_cleanup_processed_data_is_called(self, mocked_cleanup_processed_data, *_):
        main(folder='folder', dataset_name='dataset', classes=['class_1', 'class_2'])

        mocked_cleanup_processed_data.assert_called_once()


@patch('antropometria.main.DatasetReader', new=DatasetReaderMock)
@patch('antropometria.main.PreProcess')
@patch('antropometria.main.run_hyperparameter_tuning')
class TestRun:
    def test_run_works_correctly(self, mocked_run_hyperparameter_tuning, mocked_preprocess):
        run('folder', 'dataset', ['class_1', 'class_2'])

        mocked_preprocess.assert_called_once_with(dataset=dataset, labels=labels, name='folder_dataset')
        mocked_run_hyperparameter_tuning.assert_called_once_with('folder_dataset', classes_count)

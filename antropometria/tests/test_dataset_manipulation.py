from context import set_tests_context
set_tests_context()

import numpy as np
import os
import pandas as pd
import pytest

from antropometria.utils.dataset.manipulation import apply_pearson_feature_selection, combine_columns_names, \
    build_ratio_dataset, get_difference_of_classes


COLUMNS_NAMES = ['this', 'is', 'a', 'test']
DF_HIGH_CORRELATION = pd.DataFrame([[1, -1], [2, -2], [3, -3]])
DF_AVERAGE_CORRELATION = pd.DataFrame([[1, 3], [3, 1], [4, 5]])
DF_TO_CREATE_RATIO_DF = pd.DataFrame([[1, 2, 3], [5, 6, 7]])


class TestDatasetManipulation:
    def test_pearson_feature_selection_works(self):
        df_with_less_columns = apply_pearson_feature_selection(DF_HIGH_CORRELATION)
        exactly_same_df = apply_pearson_feature_selection(DF_AVERAGE_CORRELATION)

        assert DF_HIGH_CORRELATION.shape[1] > df_with_less_columns.shape[1]
        assert DF_AVERAGE_CORRELATION.equals(exactly_same_df)

    def test_pearson_feature_selection_with_invalid_threshold(self):
        with pytest.raises(ValueError):
            apply_pearson_feature_selection(pd.DataFrame([]), 2.0)

    def test_combine_columns_names_with_mode_default(self):
        combined_names = combine_columns_names(len(COLUMNS_NAMES), COLUMNS_NAMES, mode='default')

        assert len(combined_names) == (len(COLUMNS_NAMES) * (len(COLUMNS_NAMES) - 1) / 2)
        assert combined_names == ['0/1', '0/2', '0/3', '1/2', '1/3', '2/3']

    def test_combine_columns_names_with_mode_complete(self):
        combined_names = combine_columns_names(len(COLUMNS_NAMES), COLUMNS_NAMES, mode='complete')

        assert len(combined_names) == (len(COLUMNS_NAMES) * (len(COLUMNS_NAMES) - 1)) / 2
        assert combined_names == [f'{COLUMNS_NAMES[0]}/{COLUMNS_NAMES[1]}', f'{COLUMNS_NAMES[0]}/{COLUMNS_NAMES[2]}',
                                  f'{COLUMNS_NAMES[0]}/{COLUMNS_NAMES[3]}', f'{COLUMNS_NAMES[1]}/{COLUMNS_NAMES[2]}',
                                  f'{COLUMNS_NAMES[1]}/{COLUMNS_NAMES[3]}', f'{COLUMNS_NAMES[2]}/{COLUMNS_NAMES[3]}']

    def test_combine_columns_using_list_with_one_element(self):
        combined_names = combine_columns_names(1, ['only_one'])

        assert combined_names == []

    def test_build_ratio_dataset(self):
        df_ratio_name = 'new_df'
        file_path = f'./antropometria/data/ratio/{df_ratio_name}.csv'
        build_ratio_dataset(dataset=DF_TO_CREATE_RATIO_DF, name=df_ratio_name)

        assert os.path.isdir(os.path.dirname(file_path))
        assert os.path.isfile(file_path)

        df_ratio = pd.read_csv(file_path)

        assert df_ratio.shape[1] == (DF_TO_CREATE_RATIO_DF.shape[1] * (DF_TO_CREATE_RATIO_DF.shape[1] - 1)) / 2

        for row_index, row in df_ratio.iterrows():
            column_index = 0
            for column_i in range(0, row.values.size):
                value_i = DF_TO_CREATE_RATIO_DF.iloc[row_index, column_i]

                for column_j in range(column_i + 1, row.values.size):
                    value_j = DF_TO_CREATE_RATIO_DF.iloc[row_index, column_j]

                    if value_i >= value_j:
                        assert row[column_index] == np.float64(value_i / value_j)
                    else:
                        assert row[column_index] == np.float64(value_j / value_i)

                    column_index += 1

        os.remove(file_path)

    def test_get_the_difference_of_number_of_labels_in_binary_dataset(self):
        difference_of_num_of_labels = get_difference_of_classes([100, 200])

        assert difference_of_num_of_labels == 200 - 100

    def test_get_the_difference_of_number_of_labels_in_non_binary_dataset(self):
        with pytest.raises(ValueError):
            get_difference_of_classes([100])

        with pytest.raises(ValueError):
            get_difference_of_classes([1, 2, 3])

from context import set_tests_context
set_tests_context()

import pandas as pd
import pytest

from antropometria.utils.dataset.manipulation import apply_pearson_feature_selection, combine_columns_names, \
    build_ratio_dataset, get_difference_of_classes


COLUMNS_NAMES = ['this', 'is', 'a', 'test']
DF_HIGH_CORRELATION = pd.DataFrame([[1, -1], [2, -2], [3, -3]])
DF_AVERAGE_CORRELATION = pd.DataFrame([[1, 3], [3, 1], [4, 5]])
INVALID_PEARSON_THRESHOLD = 2.0


class TestDatasetManipulation:
    def test_pearson_feature_selection_works(self):
        df_with_less_columns = apply_pearson_feature_selection(DF_HIGH_CORRELATION)
        exactly_same_df = apply_pearson_feature_selection(DF_AVERAGE_CORRELATION)

        assert DF_HIGH_CORRELATION.shape[1] > df_with_less_columns.shape[1]
        assert DF_AVERAGE_CORRELATION.equals(exactly_same_df)

    def test_pearson_feature_selection_with_invalid_threshold(self):
        with pytest.raises(ValueError):
            apply_pearson_feature_selection(pd.DataFrame([]), INVALID_PEARSON_THRESHOLD)

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

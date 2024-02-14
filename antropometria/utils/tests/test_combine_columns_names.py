import pandas as pd

from antropometria.utils import combine_columns_names

COLUMNS_NAMES = pd.Index(['this', 'is', 'a', 'test'])


class TestCombineColumnsNames:
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
        combined_names = combine_columns_names(1, pd.Index(['only_one']))

        assert combined_names == []

import os
import numpy as np
import pandas as pd

from antropometria.utils import build_ratio_dataset


DF_TO_CREATE_RATIO_DF = pd.DataFrame([[1, 2, 3], [5, 6, 7]])


class TestBuildRatioDataset:
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
                        assert row.iloc[column_index] == np.float64(value_i / value_j)
                    else:
                        assert row.iloc[column_index] == np.float64(value_j / value_i)

                    column_index += 1

        os.remove(file_path)

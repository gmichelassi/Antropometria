import pandas as pd
import pytest

from antropometria.statistics import apply_pearson_feature_selection


DF_HIGH_CORRELATION = pd.DataFrame([[1, -1], [2, -2], [3, -3]])
DF_AVERAGE_CORRELATION = pd.DataFrame([[1, 3], [3, 1], [4, 5]])


class TestPearsonCorrelationFilter:
    def test_pearson_feature_selection_works(self):
        df_with_less_columns = apply_pearson_feature_selection(DF_HIGH_CORRELATION)
        exactly_same_df = apply_pearson_feature_selection(DF_AVERAGE_CORRELATION)

        assert DF_HIGH_CORRELATION.shape[1] > df_with_less_columns.shape[1]
        assert DF_AVERAGE_CORRELATION.equals(exactly_same_df)

    def test_pearson_feature_selection_with_invalid_threshold(self):
        with pytest.raises(ValueError):
            apply_pearson_feature_selection(pd.DataFrame([]), 2.0)
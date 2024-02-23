import pandas as pd
import pytest

from antropometria.statistics import PeasonCorrelationFeatureSelector

columns = ['feature1', 'feature2']

DF_HIGH_CORRELATION = pd.DataFrame([[1, -1], [2, -2], [3, -3]], columns=columns)
DF_AVERAGE_CORRELATION = pd.DataFrame([[1, 3], [3, 1], [4, 5]], columns=columns)


class TestPeasonCorrelationFeatureSelector:
    def test_pearson_correlation_feature_selector_works_correcly(self):
        df_with_less_columns = PeasonCorrelationFeatureSelector(DF_HIGH_CORRELATION).apply_pearson_feature_selection()
        exactly_same_df = PeasonCorrelationFeatureSelector(DF_AVERAGE_CORRELATION).apply_pearson_feature_selection()

        assert DF_HIGH_CORRELATION.shape[1] > df_with_less_columns.shape[1]
        assert DF_AVERAGE_CORRELATION.equals(exactly_same_df)

    def test_pearson_feature_selection_keeps_column_names(self):
        df_with_less_columns = PeasonCorrelationFeatureSelector(DF_HIGH_CORRELATION).apply_pearson_feature_selection()

        assert df_with_less_columns.columns.to_numpy().tolist() == ['feature1']

    def test_pearson_feature_selection_with_invalid_threshold(self):
        with pytest.raises(ValueError):
            PeasonCorrelationFeatureSelector(DF_HIGH_CORRELATION, threshold=2.0)

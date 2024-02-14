import pandas as pd

from antropometria.statistics import apply_min_max_normalization


DF_MIN_MAX = pd.DataFrame([[1, 2], [3, 4]])


class TestMinMaxNormalization:
    def test_applies_min_max_normalization(self):
        assert apply_min_max_normalization(DF_MIN_MAX).equals(
            pd.DataFrame([[0/3, 1/3], [2/3, 3/3]], dtype=float)
        )

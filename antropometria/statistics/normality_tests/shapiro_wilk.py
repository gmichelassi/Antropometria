import pandas as pd
from scipy.stats import shapiro


def perform_shapiro_wilk_test(data: pd.Series) -> bool:
    _, pvalue = shapiro(data.to_numpy())

    return pvalue > 0.05

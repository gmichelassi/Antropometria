import pandas as pd

from antropometria.utils import transform_string_of_numbers_into_array_of_floats
from scipy.stats import friedmanchisquare


def perform(data: pd.DataFrame, query: str, column: str, expected_amount: int = None) -> (float, float):
    filtered_data = data.query(query) if query is not '' else data

    if len(filtered_data) == 0 or (expected_amount is not None and len(filtered_data) != expected_amount):
        raise ValueError(f'When filtering data could not find the correct amount of tests ({expected_amount} tests)')

    scores = list(map(
        lambda score: transform_string_of_numbers_into_array_of_floats(score),
        filtered_data[column]
    ))

    return friedmanchisquare(*scores)

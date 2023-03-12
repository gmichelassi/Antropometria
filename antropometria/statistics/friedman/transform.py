import pandas as pd

from antropometria.config.constants import CLASSIFIER_NAMES, REDUCTIONS_NAMES, PEARSONS, MIN_MAXS, SAMPLING_NAMES
from antropometria.utils import transform_string_of_numbers_into_array_of_floats
from itertools import product

from typing import List


EXPECTED_AMOUNT = 5


def transform(data: pd.DataFrame, classifiers: List[str] = CLASSIFIER_NAMES) -> pd.DataFrame:
    df = pd.DataFrame()
    title_to_lib_and_f1_mapping = {}

    for classifier, red_dim, pearson, min_max, sampling in product(classifiers, REDUCTIONS_NAMES, PEARSONS, MIN_MAXS, SAMPLING_NAMES):
        query = f'classifier == "{classifier}" and red_dim == "{red_dim}" and pearson == {pearson} and min_max == "{min_max}" and sampling == "{sampling}"'
        filtered_data = data.query(query)

        if len(filtered_data) == 0 or len(filtered_data) != EXPECTED_AMOUNT:
            continue

        for _, test in filtered_data.iterrows():
            img_lib = test['img_lib']
            f1_score_folds = transform_string_of_numbers_into_array_of_floats(test['f1score_folds'])

            for current_fold, f1_score in enumerate(f1_score_folds):
                current_title = f'{classifier}#{red_dim}#{pearson}#{min_max}#{sampling}#{current_fold}'

                title_to_lib_and_f1_mapping[current_title] = {
                    **title_to_lib_and_f1_mapping[current_title],
                    img_lib: f1_score
                } if current_title in list(title_to_lib_and_f1_mapping.keys()) else {img_lib: f1_score}

    for row_name, values in title_to_lib_and_f1_mapping.items():
        df = pd.concat([df, pd.Series(data=values, name=row_name)], axis=1)

    return df

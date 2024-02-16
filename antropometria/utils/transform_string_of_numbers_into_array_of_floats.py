import re

PATTERN = r'\[((0|1),[0-9]*(, |))*\]'


def transform_string_of_numbers_into_array_of_floats(stringified_array: str):
    """Convert a string shaped as an array of floats in a list of floats

    param stringified_array is a string in the shape of [0.22, 0.43, ..., 0.74]
    returns a list of floats where the numbers are the same as the received ones
    """
    if not bool(re.fullmatch(PATTERN, stringified_array)):
        raise ValueError(f'Input {stringified_array} does not match required format')

    splitted_string = stringified_array[1:-1].split(', ')
    return [float(x.replace(',', '.')) for x in splitted_string]

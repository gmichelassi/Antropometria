from typing import List

from antropometria.exceptions import NonBinaryDatasetError


def get_difference_of_classes(classes_count: List[int]) -> int:
    if len(classes_count) != 2:
        raise NonBinaryDatasetError(len(classes_count))

    return abs(classes_count[0] - classes_count[1])

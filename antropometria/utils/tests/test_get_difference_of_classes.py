import pytest

from antropometria.exceptions import NonBinaryDatasetError
from antropometria.utils import get_difference_of_classes


class TestGetDifferenceOfClasses:
    def test_get_the_difference_of_number_of_labels_in_binary_dataset(self):
        difference_of_num_of_labels = get_difference_of_classes([100, 200])

        assert difference_of_num_of_labels == 200 - 100

    def test_get_the_difference_of_number_of_labels_in_non_binary_dataset(self):
        with pytest.raises(NonBinaryDatasetError):
            get_difference_of_classes([100])

        with pytest.raises(NonBinaryDatasetError):
            get_difference_of_classes([1, 2, 3])

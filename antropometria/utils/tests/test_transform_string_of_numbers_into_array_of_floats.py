import pytest

from antropometria.utils import \
    transform_string_of_numbers_into_array_of_floats


class TestTransformStringOfNumbersIntoArrayOfFloats:
    def test_works_fine(self):
        correct_input = '[0,7, 0,7, 0,666666, 0,44444444444, 0,8888888, 0,7777778, 0,55555556, 0,5556, 0,888, 0,77778]'
        expected_output = [0.7, 0.7, 0.666666, 0.44444444444, 0.8888888, 0.7777778, 0.55555556, 0.5556, 0.888, 0.77778]

        actual_output = transform_string_of_numbers_into_array_of_floats(correct_input)

        assert expected_output == actual_output

    def test_raises_exception_when_input_doesnt_match_expected_input(self):
        incorrect_input = '[0,7, 0,7, 0,6word, 0,44]'

        with pytest.raises(ValueError):
            transform_string_of_numbers_into_array_of_floats(incorrect_input)

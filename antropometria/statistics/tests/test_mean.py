import numpy as np

from antropometria.statistics import calculate_mean_from_dict


DICT = {
    'accuracy': np.random.uniform(size=(5,)),
    'precision': np.random.uniform(size=(5,)),
    'recall': np.random.uniform(size=(5,)),
    'any_metric': np.random.uniform(size=(5,))
}

METRIC = np.random.uniform(size=(5,))


class TestMetrics:
    def test_calculate_mean(self):
        result_dict = calculate_mean_from_dict(DICT)
        assert len(result_dict.keys()) == len(DICT.keys())
        assert len(result_dict.values()) == len(DICT.values())

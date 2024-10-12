import numpy as np
import pandas as pd
import os

from antropometria.config.constants import MULTICLASS_FIELDNAMES
from antropometria.utils.results import extract_results, save_results, write_header

OUTPUT_FILE = './antropometria/output/file_test.csv'
ANY = 'any'
TEST = {
    'classifier': ANY,
    'reduction': ANY,
    'filtro': ANY,
    'min_max': ANY,
    'balanceamento': ANY,
}
GRID_SEARCH_RESULTS = {
    'cv_accuracy': ANY,
    'cv_precision': ANY,
    'cv_recall': ANY,
    'cv_f1score': ANY,
}
ERROR_ESTIMATION_RESULTS = {
    'err_accuracy': ANY,
    'err_precision_micro': ANY,
    'err_recall_micro': ANY,
    'err_f1score_micro': ANY,
    'err_f1micro_ic': ANY,
    'err_precision_macro': ANY,
    'err_recall_macro': ANY,
    'err_f1score_macro': ANY,
    'err_f1macro_ic': ANY,
}
PARAMETERS = {}


class GridResults:
    def __init__(self):
        self.best_score_ = 0.9
        self.best_index_ = 0
        self.cv_results_ = {
            'mean_test_precision': [0.9, 0.1, 0.2],
            'mean_test_recall': [0.9, 0.1, 0.2],
            'mean_test_accuracy': [0.9, 0.1, 0.2],
        }
        self.best_params_ = {}
        self.best_estimator_ = 'RandomForest'


class TestParameterCalibrationResults:
    def test_get_results(self):
        accuracy, precision, recall, f1, parameters, best_estimator = extract_results(GridResults())

        assert type(accuracy) is float
        assert type(precision) is float
        assert type(recall) is float
        assert type(f1) is float
        assert type(parameters) is dict
        assert type(best_estimator) is str

    def test_save_results(self):
        save_results(
            dataset_shape=(100, 10),
            file=OUTPUT_FILE,
            fieldnames=MULTICLASS_FIELDNAMES,
            test=TEST,
            grid_search_results=GRID_SEARCH_RESULTS,
            error_estimation_results=ERROR_ESTIMATION_RESULTS,
            parameters=PARAMETERS
        )

        assert os.path.isfile(OUTPUT_FILE)
        os.remove(OUTPUT_FILE)

    def test_write_header(self):
        write_header(file=OUTPUT_FILE, fieldnames=MULTICLASS_FIELDNAMES)

        assert os.path.isfile(OUTPUT_FILE)

        header = pd.read_csv(OUTPUT_FILE).columns.to_numpy()
        assert np.array_equal(header, MULTICLASS_FIELDNAMES)

        os.remove(OUTPUT_FILE)

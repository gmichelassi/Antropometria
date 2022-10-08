import time

from antropometria.config import get_logger
from antropometria.hyperparameter_tuning import run_hyperparameter_tuning
from antropometria.preprocessing.run_preprocessing import run_preprocessing
from antropometria.utils.cleanup_processed_data import cleanup_processed_data
from antropometria.utils.load_data import LoadData
from typing import List


log = get_logger(__file__)


def main(folder: str, dataset_name: str, classes: List[str]):
    start_time = time.time()

    data_name = f'{folder}_{dataset_name}'
    x, y, classes_count = LoadData(folder, dataset_name, classes).load()

    run_preprocessing(data=(x, y), name=data_name)
    run_hyperparameter_tuning(data_name, classes_count)
    cleanup_processed_data()

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

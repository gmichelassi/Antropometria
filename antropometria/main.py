import time

from antropometria.config import get_logger
from antropometria.hyperparameter_tuning import run_hyperparameter_tuning
from antropometria.preprocessing.run_preprocessing import run_preprocessing
from antropometria.utils.load_data import LoadData
from typing import List


log = get_logger(__file__)


def main(folder: str, dataset_name: str, classes: List[str]):
    start_time = time.time()

    log.info(f'Loading data from data/{folder}/{dataset_name}')

    data = LoadData(folder, dataset_name, classes).load()

    run_preprocessing(data)

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

import time

from antropometria.config import get_logger
from antropometria.hyperparameter_tuning import run_hyperparameter_tuning
from antropometria.utils.load_data import LoadData
from typing import List


log = get_logger(__file__)


def main(folder: str, dataset_name: str, classes: List[str]):
    start_time = time.time()

    log.info(f'Loading data from data/{folder}/{dataset_name}')

    data = LoadData(folder, dataset_name, classes).load()

    run_hyperparameter_tuning('shared_instances', 'mediapipe64', ['single'])
    run_hyperparameter_tuning('shared_instances', 'opencvdnn', ['single'])
    run_hyperparameter_tuning('shared_instances', 'opencvhaar', ['single'])
    run_hyperparameter_tuning('shared_instances', 'openface', ['single'])
    run_hyperparameter_tuning('shared_instances', 'mediapipecustom', ['single'])

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

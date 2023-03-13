import time

from antropometria.config import get_logger
from antropometria.hyperparameter_tuning import run_hyperparameter_tuning
from antropometria.preprocessing.run_preprocessing import run_preprocessing
from antropometria.utils.cleanup_processed_data import cleanup_processed_data
from antropometria.utils.load_data import LoadData
from typing import List


log = get_logger(__file__)

DEFAULT_VALUES = [
    ('shared_instances', 'dlibcnn',         ['1']),
    ('shared_instances', 'opencvdnn',       ['2']),
    ('shared_instances', 'openface',        ['3']),
    ('shared_instances', 'mediapipe64',     ['4']),
    ('shared_instances', 'mediapipecustom', ['5']),
]


def run(folder: str, dataset_name: str, classes: List[str]):
    data_name = f'{folder}_{dataset_name}'
    x, y, classes_count = LoadData(folder, dataset_name, classes).load()

    run_preprocessing(data=(x, y), name=data_name)
    run_hyperparameter_tuning(data_name, classes_count)


def main(use_default_values: bool = True, **kwargs):
    start_time = time.time()
    if use_default_values:
        for folder, dataset_name, classes in DEFAULT_VALUES:
            run(folder, dataset_name, classes)
    else:
        folder, dataset_name, classes = tuple(kwargs.values())
        run(folder, dataset_name, classes)

    cleanup_processed_data()

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

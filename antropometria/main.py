import time

from antropometria.config import get_logger
from antropometria.data import DatasetReader
from antropometria.hyperparameter_tuning import run_hyperparameter_tuning
from antropometria.preprocessing import PreProcess
from antropometria.utils.cleanup_processed_data import cleanup_processed_data


log = get_logger(__file__)


def run(folder: str, dataset_name: str, classes: list[str]):
    data_name = f'{folder}_{dataset_name}'

    reader = DatasetReader(folder=folder, dataset_name=dataset_name, classes=classes)
    x, y, classes_count = reader.read()

    preprocessing = PreProcess(dataset=x, labels=y, name=data_name)
    preprocessing.run()
    run_hyperparameter_tuning(data_name, classes_count)


def main(*args, **kwargs):
    start_time = time.time()

    if 'data' in kwargs:
        for folder, dataset_name, classes in kwargs['data']:
            run(folder, dataset_name, classes)
    elif 'folder' in kwargs and 'dataset_name' in kwargs and 'classes' in kwargs:
        folder: str = kwargs.get('folder')
        dataset_name: str = kwargs.get('dataset_name')
        classes: list[str] = kwargs.get('classes')

        run(folder, dataset_name, classes)
    else:
        raise ValueError(f"Invalid arguments {args} and {kwargs}")

    cleanup_processed_data()

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

import time

from antropometria.config import get_logger
from antropometria.hyperparameter_tuning import run_hyperparameter_tuning


log = get_logger(__file__)


def main():
    start_time = time.time()

    run_hyperparameter_tuning('shared_instances', 'mediapipe64', ['single'])
    run_hyperparameter_tuning('shared_instances', 'opencvdnn', ['single'])
    run_hyperparameter_tuning('shared_instances', 'opencvhaar', ['single'])
    run_hyperparameter_tuning('shared_instances', 'openface', ['single'])
    run_hyperparameter_tuning('shared_instances', 'mediapipecustom', ['single'])

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

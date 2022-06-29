import time

from antropometria.config import get_logger
from antropometria.hyperparameter_tuning import run_hyperparameter_tuning


log = get_logger(__file__)


def main():
    start_time = time.time()

    run_hyperparameter_tuning('mediapipe64', 'distances_px', ['casos', 'controles'])
    run_hyperparameter_tuning('opencvdnn', 'distances_px', ['casos', 'controles'])
    run_hyperparameter_tuning('opencvhaar', 'distances_px', ['casos', 'controles'])
    run_hyperparameter_tuning('openface', 'distances_px', ['casos', 'controles'])
    run_hyperparameter_tuning('mediapipecustom', 'distances_px', ['casos', 'controles'])

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

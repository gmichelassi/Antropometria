import time

from antropometria.config import get_logger
from antropometria.hyperparameter_tuning import run_hyperparameter_tuning


log = get_logger(__file__)


def main():
    start_time = time.time()

    run_hyperparameter_tuning('openface', 'distances_px', ['casos', 'controles'])

    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

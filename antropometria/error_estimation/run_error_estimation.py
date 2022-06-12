import time

from antropometria.config import ERROR_ESTIMATION
from antropometria.config import logger


log = logger.get_logger(__file__)


def run_error_estimation(x, y, classes_count, best_estimator, sampling, verbose):
    log.info(f'Running error estimation')

    initial_time = time.time()

    error_estimation_strategy = ERROR_ESTIMATION[str(sampling)]
    error_estimation_results = error_estimation_strategy(x, y, classes_count, best_estimator).run_error_estimation()

    ellapsed_time = (time.time() - initial_time) / 60

    log.info(f"Finished error estimation in {ellapsed_time:.2f} minutes") if verbose else lambda: None

    return error_estimation_results

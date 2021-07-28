import logging
import os

from constants import ROOT_DIR


def get_logger(logged_file: str = __file__):
    logger = logging.getLogger(os.path.basename(logged_file))
    if not logger.handlers:
        logging.basicConfig(
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%d/%m/%Y %I:%M:%S %p',
            filename=ROOT_DIR + "/antropometria/output/processing.log",
            filemode='w',
            level=logging.DEBUG
        )
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s',
            datefmt='%d/%m/%Y %I:%M:%S'
        )
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

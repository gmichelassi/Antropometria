import logging
import os

ROOT_DIR = os.path.abspath(os.getcwd())


def get_logger(logger_for_file):
    logger = logging.getLogger(os.path.basename(logger_for_file))
    if not logger.handlers:
        logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S %p', filename=ROOT_DIR + "/output/processing.log",
                            filemode='w', level=logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s',
                                      datefmt='%d/%m/%Y %I:%M:%S')
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

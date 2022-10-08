import os
import shutil

from antropometria.config.constants import PROCESSED_DIR


def cleanup_processed_data():
    if not os.path.exists(PROCESSED_DIR):
        return

    shutil.rmtree(PROCESSED_DIR)

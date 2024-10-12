import os
import shutil

import pytest

from antropometria.config import PROCESSED_DIR
from antropometria.utils.cleanup_processed_data import cleanup_processed_data


@pytest.fixture
def setup_directory():
    if not os.path.exists(PROCESSED_DIR):
        os.mkdir(PROCESSED_DIR)

    for i in range(10):
        with open(f'{PROCESSED_DIR}/file_{i}.txt', 'w') as f:
            f.write(f'file_{i}')

    yield

    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)


class TestCleanupProcessedData:
    def test_cleanup_processed_data(self, setup_directory):
        assert os.path.exists(PROCESSED_DIR)

        cleanup_processed_data()

        assert not os.path.exists(PROCESSED_DIR)

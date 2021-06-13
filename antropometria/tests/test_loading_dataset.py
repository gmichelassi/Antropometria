from tests_context import set_tests_context
set_tests_context()

import pytest
import numpy as np

from antropometria.utils.dataset.load import LoadData


DATASET_MULTIPLE_FILES = 'distances_all_px_eu'
DATASET_SINGLE_FILE = 'wine'
FOLDER_MULTIPLE_FILES = 'dlibHOG'
FOLDER_SINGLE_FILE = 'dumb'
EMPTY_CLASSES = []
MULTIPLE_CLASSES = ['casos', 'controles']
SINGLE_CLASS = ['single_class']


class TestLoadingDataset:
    def test_loading_without_classes(self):
        with pytest.raises(IOError):
            LoadData(DATASET_MULTIPLE_FILES, DATASET_MULTIPLE_FILES, EMPTY_CLASSES).load()

    def test_loading_dataset_from_multiple_files(self):
        x, y = LoadData(FOLDER_MULTIPLE_FILES, DATASET_MULTIPLE_FILES, MULTIPLE_CLASSES).load()

        assert len(np.unique(y)) == len(MULTIPLE_CLASSES)
        assert x.shape[0] == len(y)

    def test_loading_dataset_from_single_file(self):
        x, y = LoadData(FOLDER_SINGLE_FILE, DATASET_SINGLE_FILE, SINGLE_CLASS).load()

        assert x.shape[0] == len(y)

    def test_loading_non_existing_dataset(self):
        with pytest.raises(IOError):
            LoadData(FOLDER_SINGLE_FILE, f"{DATASET_SINGLE_FILE}_DONOTEXIST", SINGLE_CLASS).load()

        with pytest.raises(IOError):
            LoadData(FOLDER_MULTIPLE_FILES, f"{DATASET_MULTIPLE_FILES}_DONOTEXIST", SINGLE_CLASS).load()

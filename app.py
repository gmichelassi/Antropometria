import argparse
import os

from antropometria import main
from antropometria.config import DATA_DIR

FOLDER_CHOICES = [file for file in os.listdir(DATA_DIR) if os.path.isdir(f'{DATA_DIR}{file}')]
DEFAULT_VALUES = [
    ('dumb', 'wine', None),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Machine Learning Pipeline')
    parser.add_argument('--folder', choices=FOLDER_CHOICES, help='The folder name inside \'antropometria/data/\'')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument(
        '-c', '--classes', action='append', help='The classes of your data. One argument for each class.', default=[]
    )

    args = parser.parse_args()

    folder, dataset_name, classes = args.folder, args.dataset, args.classes

    if folder is None or dataset_name is None:
        main(data=DEFAULT_VALUES)
    else:
        main(folder=folder, dataset_name=dataset_name, classes=classes)

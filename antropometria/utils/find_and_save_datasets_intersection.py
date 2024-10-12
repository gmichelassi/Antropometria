from functools import reduce
from typing import List

from antropometria.data import DatasetReader


def find_and_save_datasets_intersection(datasets_name: List[str], key_column: str):
    all_instances = []
    all_data = []

    for dataset in datasets_name:
        reader = DatasetReader(folder='shared_instances', dataset_name=dataset, classes=['1'])
        x, _, _ = reader.read()

        all_data.append(x)
        all_instances.append(x[key_column].to_list())

    instances_in_all_datasets = list(reduce(set.intersection, [set(item) for item in all_instances]))
    intersection_amount = len(instances_in_all_datasets)

    print(f'Intersection contained: {intersection_amount} instances')

    for dataset, dataset_name in zip(all_data, datasets_name):
        filtered_dataset = dataset.loc[dataset[key_column].isin(instances_in_all_datasets)].drop([key_column], axis=1)

        print(f'Processed {dataset_name} with {intersection_amount} instances.')

        filtered_dataset.to_csv(f'./{dataset_name}-{intersection_amount}-shared-instances.csv', index=False)

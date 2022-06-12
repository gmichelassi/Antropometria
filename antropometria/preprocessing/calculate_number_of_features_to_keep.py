import numpy as np

from sklearn.decomposition import PCA


def calculate_number_of_neatures_to_keep(
        original_number_of_features: int,
        current_number_of_features: int,
        dataset: np.ndarray
) -> int:
    sqrt_original_features = int(np.sqrt(original_number_of_features))

    if sqrt_original_features < current_number_of_features:
        return current_number_of_features

    pca = PCA()
    pca.fit(dataset)

    threshold = 0.9
    accumulated_variance = 0
    number_of_relevant_features = 0
    for explained_variance in pca.explained_variance_ratio_:
        if accumulated_variance > threshold:
            break

        accumulated_variance = accumulated_variance + explained_variance
        number_of_relevant_features = number_of_relevant_features + 1

    return number_of_relevant_features \
        if (number_of_relevant_features / current_number_of_features) >= 0.6 else current_number_of_features

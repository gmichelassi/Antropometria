from antropometria.config.types import Classifier, Reduction, Sampling


def map_test_to_dict(
        dataset: str,
        classifier: Classifier,
        reduction: Reduction,
        p_filter: float,
        min_max: bool,
        sampling: Sampling
) -> dict:
    return {
        'dataset': dataset,
        'classifier': classifier,
        'reduction': reduction,
        'filtro': p_filter,
        'min_max': min_max,
        'balanceamento': sampling,
    }


def map_grid_search_results_to_dict(accuracy: float, precision: float, recall: float, f1: float) -> dict:
    return {
        'cv_accuracy': accuracy,
        'cv_precision': precision,
        'cv_recall': recall,
        'cv_f1score': f1,
    }

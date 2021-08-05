def test_to_dict(
        folder: str,
        model_name: str,
        reduction: str,
        p_filter: float,
        min_max: bool,
        sampling: str
) -> dict:
    return {
        'biblioteca': folder,
        'classifier': model_name,
        'reduction': reduction,
        'filtro': p_filter,
        'min_max': min_max,
        'balanceamento': sampling,
    }


def grid_search_results_to_dict(accuracy: float, precision: float, recall: float, f1: float) -> dict:
    return {
        'cv_accuracy': accuracy,
        'cv_precision': precision,
        'cv_recall': recall,
        'cv_f1score': f1,
    }
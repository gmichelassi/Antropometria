from antropometria.classifiers.RandomForests import RandomForests
from typing import Any, Union


def stop_running_rf(
        is_random_forest_done: bool = False,
        classifier_name: str = 'Any',
        reduction: Union[str, None] = 'Any'
) -> bool:
    if not is_random_forest_done and classifier_name == 'RandomForestClassifier' and reduction is None:
        return False
    if not is_random_forest_done and classifier_name == 'RandomForestClassifier' and reduction == 'RFSelect':
        return True
    if is_random_forest_done:
        return True

    return False


def skip_current_test(
        is_random_forest_done: bool = False,
        classifier: Any = 'Any',
        reduction: Union[str, None] = 'Any'
) -> bool:
    if is_random_forest_done and classifier == RandomForests:
        return True
    if classifier != RandomForests and reduction is None:
        return True

    return False

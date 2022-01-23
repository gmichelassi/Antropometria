from typing import Union


def skip_current_test(
        classifier: str = 'Any',
        reduction: Union[str, None] = 'Any'
) -> bool:

    print(classifier, reduction)
    print(reduction in [None])
    if classifier == 'RandomForest' and reduction in ['RFSelect', None]:
        return False

    if classifier != 'RandomForest' and reduction is None:
        return True

    return True


from typing import Union


def skip_current_test(
        classifier: str = 'Any',
        reduction: Union[str, None] = 'Any'
) -> bool:
    if classifier == 'RandomForests' and reduction not in ['RFSelect', None]:
        return True

    if classifier != 'RandomForests' and reduction is None:
        return True

    return False


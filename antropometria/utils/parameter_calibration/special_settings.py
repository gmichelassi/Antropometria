from antropometria.config.types import Classifier, Reduction
from typing import Optional


def skip_current_test(
        classifier: Optional[Classifier],
        reduction: Optional[Reduction]
) -> bool:
    if classifier == 'RandomForest' and reduction not in ['RFSelect', None]:
        return True

    if classifier != 'RandomForest' and reduction is None:
        return True

    return False


def stop_running_rf(is_random_forest_done, classifier_name, reduction):
    if not is_random_forest_done and classifier_name == 'RandomForestClassifier' and reduction is None:
        return False
    if not is_random_forest_done and classifier_name == 'RandomForestClassifier' and reduction == 'RFSelect':
        return True
    if is_random_forest_done:
        return True

    return False


def skip_current_test(is_random_forest_done, classifier, reduction, rf):
    if is_random_forest_done and classifier == rf:
        return True
    if classifier != rf and reduction is None:
        return True

    return False

import math
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from antropometria.utils.metrics import calculate_mean, calculate_std
from antropometria.utils.dataset.manipulation import get_difference_of_classes
from typing import Any


N_SPLITS = 10
SCORING = ['accuracy', 'precision', 'recall', 'f1']
CV = StratifiedKFold(n_splits=N_SPLITS)


def complete_frame(
        x: np.ndarray,
        y: np.ndarray,
        synthetic_x: np.ndarray,
        synthetic_y: np.ndarray,
        current_fold: int
) -> tuple[np.ndarray, np.ndarray]:
    synthetic_x = np.array_split(synthetic_x, N_SPLITS)
    synthetic_y = np.array_split(synthetic_y, N_SPLITS)

    for i in range(len(synthetic_x)):
        if i != current_fold:
            for j in range(len(synthetic_x[i])):
                x = np.append(arr=x, values=[synthetic_x[i][j]], axis=0)
                y = np.append(arr=y, values=[synthetic_y[i][j]], axis=0)

    return x, y


def error_estimation(
        x: np.ndarray,
        y: np.ndarray,
        classes_count: list,
        estimator: Any
) -> dict:
    n = get_difference_of_classes(classes_count)

    if n == 0:
        return {
            'err_accuracy': '-', 'err_IC': '-', 'err_precision_micro': '-', 'err_recall_micro': '-',
            'err_f1score_micro': '-', 'err_precision_macro': '-', 'err_recall_macro': '-', 'err_f1score_macro': '-'
        }

    synthetic_x = x[-n:]
    synthetic_y = y[-n:]
    x = x[:-n]
    y = y[:-n]

    current_fold = 0
    folds = []
    for train_index, test_index in CV.split(x, y):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        x_train, y_train = complete_frame(x_train, y_train, synthetic_x, synthetic_y, current_fold)
        folds.append((x_train, y_train, x_test, y_test))
        current_fold += 1

    accuracy = []
    precision_micro = []
    recall_micro = []
    f1_micro = []
    precision_macro = []
    recall_macro = []
    f1_macro = []

    for i in range(N_SPLITS):
        estimator.fit(folds[i][0], folds[i][1])
        y_predict = estimator.predict(folds[i][2])

        accuracy.append(accuracy_score(y_true=folds[i][3], y_pred=y_predict))

        precision_micro.append(precision_score(y_true=folds[i][3], y_pred=y_predict, average='micro'))
        recall_micro.append(recall_score(y_true=folds[i][3], y_pred=y_predict, average='micro'))
        f1_micro.append(f1_score(y_true=folds[i][3], y_pred=y_predict, average='micro'))

        precision_macro.append(precision_score(y_true=folds[i][3], y_pred=y_predict, average='macro'))
        recall_macro.append(recall_score(y_true=folds[i][3], y_pred=y_predict, average='macro'))
        f1_macro.append(f1_score(y_true=folds[i][3], y_pred=y_predict, average='macro'))

    mean_results = calculate_mean({'accuracy': accuracy,
                                   'precision_micro': precision_micro,
                                   'recall_micro': recall_micro,
                                   'f1_micro': f1_micro,
                                   'precision_macro': precision_macro,
                                   'recall_macro': recall_macro,
                                   'f1_macro': f1_macro})

    tc = 2.262
    std = calculate_std(accuracy)
    ic_upper = mean_results['accuracy'] + tc * (std / math.sqrt(N_SPLITS))
    ic_lower = mean_results['accuracy'] - tc * (std / math.sqrt(N_SPLITS))

    ic = (ic_lower, ic_upper)

    return {'err_accuracy': mean_results['accuracy'],
            'err_IC': ic,
            'err_precision_micro': mean_results['precision_micro'],
            'err_recall_micro': mean_results['recall_micro'],
            'err_f1score_micro': mean_results['f1_micro'],
            'err_precision_macro': mean_results['precision_macro'],
            'err_recall_macro': mean_results['recall_macro'],
            'err_f1score_macro': mean_results['f1_macro']}

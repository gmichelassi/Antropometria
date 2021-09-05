import os

from sklearn.model_selection import StratifiedKFold

N_SPLITS = 10

CV = StratifiedKFold(n_splits=N_SPLITS)
MULTICLASS_FIELDNAMES = [
    'biblioteca',
    'classifier',
    'reduction',
    'filtro',
    'min_max',
    'balanceamento',
    'cv_accuracy',
    'cv_precision',
    'cv_recall',
    'cv_f1score',
    'err_accuracy',
    'err_precision_micro',
    'err_recall_micro',
    'err_f1score_micro',
    'err_f1micro_ic',
    'err_precision_macro',
    'err_recall_macro',
    'err_f1score_macro',
    'err_f1macro_ic',
    'parameters',
]

BINARY_FIELDNAMES = [
    'biblioteca',
    'classifier',
    'reduction',
    'filtro',
    'min_max',
    'balanceamento',
    'cv_accuracy',
    'cv_precision',
    'cv_recall',
    'cv_f1score',
    'err_accuracy',
    'err_precision',
    'err_recall',
    'err_f1score',
    'err_f1_ic',
    'parameters',
]
ROOT_DIR = os.path.abspath(os.getcwd())
TEMPORARY_RANDOM_SAMPLES = 'antropometria/output/temp_removed_random_samples.csv'
TEMPORARY_RANDOM_SAMPLES_LABELS = 'antropometria/output/temp_removed_random_samples_labels.csv'

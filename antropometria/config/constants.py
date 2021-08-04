import os

from sklearn.model_selection import StratifiedKFold

N_SPLITS = 10

CV = StratifiedKFold(n_splits=N_SPLITS)
EMPTY = '-'
FIELDNAMES = [
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
FILTERS = [0.0]
MIN_MAX_NORMALIZATION = [False, True]
REDUCTIONS = [None, 'PCA', 'mRMR', 'FCBF', 'CFS', 'RFS', 'ReliefF', 'RFSelect']
ROOT_DIR = os.path.abspath(os.getcwd())
SAMPLINGS = ['Random', 'Smote', 'Borderline', 'KMeans', 'SVM', 'Tomek']
SCORING = ['accuracy', 'precision', 'recall', 'f1']
TEMPORARY_RANDOM_SAMPLES = 'antropometria/output/temp_removed_random_samples.csv'
TEMPORARY_RANDOM_SAMPLES_LABELS = 'antropometria/output/temp_removed_random_samples_labels.csv'
EMPTY_ERROR_ESTIMATION_DICT = {
    'err_accuracy': EMPTY,
    'err_precision_micro': EMPTY,
    'err_recall_micro': EMPTY,
    'err_f1score_micro': EMPTY,
    'err_f1micro_ic': EMPTY,
    'err_precision_macro': EMPTY,
    'err_recall_macro': EMPTY,
    'err_f1score_macro': EMPTY,
    'err_f1macro_ic': EMPTY,
}

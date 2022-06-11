from antropometria.utils.error_estimation.DefaultErrorEstimation import DefaultErrorEstimation
from antropometria.utils.error_estimation.RandomSamplingErrorEstimation import RandomSamplingErrorEstimation
from antropometria.utils.error_estimation.SmoteErrorEstimation import SmoteErrorEstimation

BINARY = True
ERROR_ESTIMATION = {
    'None': DefaultErrorEstimation,
    'Random': RandomSamplingErrorEstimation,
    'Smote': SmoteErrorEstimation,
    'Borderline': SmoteErrorEstimation,
    'KMeans': SmoteErrorEstimation,
    'SVM': SmoteErrorEstimation,
    'Tomek': SmoteErrorEstimation
}
FILTERS = [0.98]
MIN_MAX_NORMALIZATION = [True]
REDUCTIONS = [None, 'PCA', 'ReliefF', 'RFSelect', 'mRMR', 'FCBF', 'CFS', 'RFS']
SAMPLINGS = [None, 'Random', 'Smote', 'Borderline', 'KMeans', 'SVM', 'Tomek']
SCORING = ['accuracy', 'precision', 'recall', 'f1']

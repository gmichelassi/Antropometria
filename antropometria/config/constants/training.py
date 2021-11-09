from antropometria.utils.error_estimation.DefaultErrorEstimation import DefaultErrorEstimation
from antropometria.utils.error_estimation.RandomSamplingErrorEstimation import RandomSamplingErrorEstimation
from antropometria.utils.error_estimation.SmoteErrorEstimation import SmoteErrorEstimation


ERROR_ESTIMATION = {
    'None': DefaultErrorEstimation,
    'Random': RandomSamplingErrorEstimation,
    'Smote': SmoteErrorEstimation,
    'Borderline': SmoteErrorEstimation,
    'KMeans': SmoteErrorEstimation,
    'SVM': SmoteErrorEstimation,
    'Tomek': SmoteErrorEstimation
}
FILTERS = [0.0, 0.98, 0.99]
MIN_MAX_NORMALIZATION = [False, True]
REDUCTIONS = [None, 'PCA', 'ReliefF', 'RFSelect', 'mRMR', 'FCBF', 'CFS', 'RFS']
SAMPLINGS = [None, 'Random', 'Smote', 'Borderline', 'KMeans', 'SVM', 'Tomek']
SCORING = ['accuracy', 'precision', 'recall', 'f1']

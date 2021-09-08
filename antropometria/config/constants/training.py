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
FILTERS = [0.0]
MIN_MAX_NORMALIZATION = [False, True]
REDUCTIONS = ['ReliefF', 'RFSelect', 'RFS']  # None, 'PCA', 'mRMR', 'FCBF', 'CFS', 'RFS', , 'RFSelect'
SAMPLINGS = [None, 'Random', 'Smote', 'Borderline', 'KMeans', 'SVM', 'Tomek']
SCORING = ['accuracy', 'precision', 'recall', 'f1']

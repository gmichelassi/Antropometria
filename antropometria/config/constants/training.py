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
FILTERS = [0.0, 0.99, 0.98]
MIN_MAX_NORMALIZATION = [False, True]
REDUCTIONS = ['FCBF', 'RFS', 'RFSelect']  # None, 'PCA', 'mRMR', , 'CFS', , 'ReliefF',
SAMPLINGS = ['KMeans', 'SVM', 'Tomek']  # None, 'Random', 'Smote', 'Borderline',
SCORING = ['accuracy', 'precision', 'recall', 'f1']

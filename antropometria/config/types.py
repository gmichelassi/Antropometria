from typing import Union

Classifier = Union[
    'KNeighborsClassifier',
    'NaiveBayes',
    'NeuralNetwork',
    'RandomForest',
    'SupportVectorMachine',
    'LinearDiscriminantAnalysis'
]
Reduction = Union['PCA', 'ReliefF', 'RFSelect', 'mRMR', 'FCBF', 'CFS', 'RFS']
Sampling = Union['Random', 'Smote', 'Borderline', 'KMeans', 'SVM', 'Tomek']

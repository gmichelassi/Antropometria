from antropometria.classifiers import (
    KNearestNeighbors as Knn,
    NaiveBayes as Nb,
    NeuralNetwork as Nn,
    RandomForest as Rf,
    SupportVectorMachine as Svm
)


BINARY = True
CLASSIFIERS = [Svm, Nn, Rf, Knn, Nb]

FILTERS = [0.98]
MIN_MAX_NORMALIZATION = [True]
REDUCTIONS = [None, 'PCA', 'ReliefF', 'RFSelect', 'mRMR', 'FCBF', 'CFS', 'RFS']
SAMPLINGS = [None, 'Random', 'Smote', 'Borderline', 'KMeans', 'SVM', 'Tomek']
SCORING = ['accuracy', 'precision', 'recall', 'f1']

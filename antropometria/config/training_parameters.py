from antropometria.classifiers import (
    KNearestNeighbors as Knn,
    NaiveBayes as Nb,
    NeuralNetwork as Nn,
    RandomForest as Rf,
    SupportVectorMachine as Svm,
    DiscriminantAnalysis as Lda
)


BINARY = True
CLASSIFIERS = [Knn, Nb, Nn, Rf, Svm, Lda]

FILTERS = [0.98]
MIN_MAX_NORMALIZATION = [False, True]
REDUCTIONS = [None, 'PCA', 'ReliefF', 'RFSelect', 'mRMR', 'FCBF', 'CFS', 'RFS']
SAMPLINGS = [None, 'Random', 'Smote', 'Borderline', 'KMeans', 'SVM', 'Tomek']
SCORING = ['accuracy', 'precision', 'recall', 'f1']

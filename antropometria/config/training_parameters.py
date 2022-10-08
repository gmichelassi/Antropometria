from antropometria.classifiers import (
    KNearestNeighbors as Knn,
    NaiveBayes as Nb,
    NeuralNetwork as Nn,
    RandomForest as Rf,
    SupportVectorMachine as Svm,
    LinearDiscriminantAnalysis as Lda
)


BINARY = True
CLASSIFIERS = [Svm, Nn, Rf, Knn, Nb, Lda]

FILTERS = [0.0]
MIN_MAX_NORMALIZATION = [False]
REDUCTIONS = [None]
SAMPLINGS = [None]
SCORING = ['accuracy', 'precision', 'recall', 'f1']

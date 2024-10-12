from antropometria.classifiers import DiscriminantAnalysis as Lda
from antropometria.classifiers import KNearestNeighbors as Knn
from antropometria.classifiers import NaiveBayes as Nb
from antropometria.classifiers import NeuralNetwork as Nn
from antropometria.classifiers import RandomForest as Rf
from antropometria.classifiers import SupportVectorMachine as Svm

BINARY = True
CLASSIFIERS = [Knn, Nb, Nn, Rf, Svm, Lda]

FILTERS = [0.98]
MIN_MAX_NORMALIZATION = [False, True]
REDUCTIONS = [None, 'PCA', 'ReliefF', 'RFSelect', 'mRMR', 'FCBF', 'CFS', 'RFS']
SAMPLINGS = [None, 'Random', 'Smote', 'Borderline', 'KMeans', 'SVM', 'Tomek']
SCORING = ['accuracy', 'precision', 'recall', 'f1']

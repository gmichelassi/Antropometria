from typing import Union, NewType

Classifier = Union['KNeighborsClassifier', 'NaiveBayes', 'NeuralNetwork', 'RandomForest', 'SupportVectorMachine']
Reduction = Union['PCA', 'ReliefF', 'RFSelect', 'mRMR', 'FCBF', 'CFS', 'RFS']
Sampling = Union['Random', 'Smote', 'Borderline', 'KMeans', 'SVM', 'Tomek']

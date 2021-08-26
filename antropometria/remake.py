import math
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.model_selection import train_test_split
import antropometria.utils.error_estimation.ErrorEstimation
import antropometria.utils.error_estimation.ErrorEstimation
from antropometria.classifiers.KNearestNeighbors import KNearestNeighbors as Knn
from antropometria.classifiers.NaiveBayes import NaiveBayes as Nb
from antropometria.classifiers.NeuralNetwork import NeuralNetwork as Nn
from antropometria.classifiers.RandomForests import RandomForests as Rf
from antropometria.classifiers.SupportVectorMachine import SupportVectorMachine as Svm
from antropometria.config import logger  # daqui pra baixo coloca antropometria
from antropometria.feature_selectors.utils.getter import get_feature_selector
from antropometria.sampling.OverSampling import OverSampling
from antropometria.sampling.UnderSampling import UnderSampling
from antropometria.utils.dataset.load import LoadData

log = logger.get_logger(__file__)


def main(metrica: str, numero: int, nfeatures: int = 0):
    dataset, Y = LoadData('dlibHOG', 'distances_sem_boca_px_eu', ["casos", "controles"]).load()

    balanceadores = ['KMeans']  #'Borderline' None'Random',,'Tomek' ,  'SVM', ,,'Smote'None-o
    redutores = [ 'PCA' ]  #'mRMR''RFS''RFSelect',None ,'FCBF',, ,  ,                 CFS''ReliefF'
    indutores = [Svm]     #Nb, Knn,,Nn ,, ,          Rf

    CV = StratifiedKFold(n_splits=numero)

    X = dataset.to_numpy()

    splits = CV.get_n_splits(X, Y)

    if nfeatures == 0:
        nfeatures = int(math.sqrt(X.shape[1]))

    VC = CV.split(X,Y)

    NFS_total =[]
    PCA_features = []


    all_X = np.ndarray([0,0])
    all_Y = np.ndarray([0,0])
    for train_index, test_index in VC:
        X_aux, Y_aux = X[train_index],Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index]

        X_train, X_calibration, Y_train, Y_calibration = train_test_split(X_aux, Y_aux, stratify=Y_aux, train_size=0.89)

        all_X = np.append(all_X, [X_train, X_test, X_calibration])
        all_Y = np.append(all_Y,[Y_train, Y_test, Y_calibration])



    for balanceador in balanceadores:
        for redutor in redutores:
            i=0
            while i < all_X.shape[0]:

                x = all_X[i]
                y = all_Y[i]

                if redutor == 'PCA':
                    aux = preprocessing_splits(x,y,balanceador,redutor)
                    PCA_features.append(aux)
                else:
                    aux = preprocessing_splits(x, y, balanceador, redutor)
                    NFS_total.append(aux)

                i = 3+i
    non_redundant_feature_sets= best_reductions(NFS_total, nfeatures)
    non_redundant_feature_sets_PCA = best_reductions(PCA_features, nfeatures)

    for indutor in indutores:
        for balanceador in balanceadores:
            m =0
            log.info(f'running{indutor} {balanceador}')
            for n in range(len(non_redundant_feature_sets) + len(PCA_features)):
                if(n<len(non_redundant_feature_sets)):
                    reductions = non_redundant_feature_sets[n]

                els
                    reductions = PCA_features[m]e:
                features = len(reductions)
                print(features)
                model = indutor(n_features=features)
                estimator = model.estimator
                media_ideal = 0.0

                for parametros in ParameterGrid(model.parameter_grid):
                    soma = 0.0
                    i =0
                    while i < all_X.shape[0]:
                        X_train, X_test, X_calibration = all_X[i], all_X[i+1], all_X[i+2]
                        Y_train, Y_test, Y_calibration = all_Y[i], all_Y[i+1], all_Y[i+2]

                        if balanceador != None:
                            if balanceador == "Random":
                                X_train, Y_train = UnderSampling().fit_transform(X_train, Y_train)
                            else:
                                X_train, Y_train = OverSampling(balanceador).fit_transform(X_train, Y_train)
                        X_train = apply_reductions(X_train,reductions)
                        X_calibration = apply_reductions(X_calibration,reductions)

                        estimator.fit(X_train, Y_train)
                        Y_predict = estimator.predict(X_calibration)


                        if (metrica == "f1"):
                            f1 = f1_score(Y_calibration, Y_predict)  # media de todos os f1 scor
                            soma += f1
                        elif (metrica == "accuracy"):
                            acuracia = accuracy_score(Y_calibration, Y_predict)
                            soma += acuracia

                        elif (metrica == 'precision'):
                            precisao = precision_score(Y_calibration, Y_predict)
                            soma += precisao
                        else:
                            recall = recal_score(Y_calibration, Y_predict)
                            soma += recall
                        i=i+3
                    media = soma / splits
                    if media > media_ideal:
                        media_ideal = media
                    parametros_otimos = parametros
                print(media_ideal)


                media_otima = 0
                soma = 0
                i =0
                while i < all_X.shape[0]:
                    X_train, X_test = np.concatenate((all_X[i],all_X[i+2]), axis =0), all_X[i + 1]
                    Y_train, Y_test = np.concatenate((all_Y[i],all_Y[i+2]), axis =0), all_Y[i + 1]


                    if balanceador == "Random":
                        X_train, Y_train = UnderSampling().fit_transform(X, Y)
                    else:
                        X_train, Y_train = OverSampling(balanceador).fit_transform(X, Y)

                    apply_reductions(X_train, non_redundant_feature_sets[n])
                    apply_reductions(X_test, non_redundant_feature_sets[n])

                    estimator.fit(X_train, Y_train)
                    Y_test_predict = estimator.predict(X_test)
                    if (metrica == "f1"):
                        f1 = f1_score(Y_test, Y_test_predict)  # media de todos os f1 score
                        soma += f1
                    elif (metrica == "accuracy"):
                        acuracia = accuracy_score(Y_test, Y_test_predict)
                        soma += acuracia
                    elif (metrica == 'precision'):
                        precisao = precision_score(Y_test, Y_test_predict)
                        soma += precisao
                    else:
                        recall = recal_score(Y_test, Y_test_predict)
                        soma += recall
                    i = i+3
                m = m+1
                media_otima = soma / (splits)
                print(media_otima)


def apply_reductions(X: np.ndarray, characteristics):
    if (type(characteristics[0]) == int):
        deletar = []
        for i in range(X.shape[1]):
            if i not in characteristics:
                deletar.append(i)
        X = np.delete(X, deletar, axis = 1)
        return X
    else:
        aux = pd.DataFrame(data= X)
        aux = aux.T

        auxPandas = pd.DataFrame(characteristics,columns = aux.columns )
        print(auxPandas.shape)
        auxPandas = auxPandas.T
        return auxPandas.to_numpy()

def best_reductions(characteristics, nfeatures):
    non_redundant_features_sets = []
    for i in characteristics:
        if i not in non_redundant_features_sets:
            non_redundant_features_sets.append(i)
    if(type(characteristics) == int):
        numpyarr = np.array(characteristics)
        flat = numpyarr.flatten()
        consensus_features= np.bincount(flat)
        if consensus_features[0:nfeatures] not in non_redundant_features_sets:
            non_redundant_features_sets.append(consensus_features[0:features])
    else:
        return non_redundant_features_sets
    return non_redundant_features_sets


def preprocessing_splits(X:np.ndarray , Y:np.ndarray, sampling: str = None, selector: str = None):
    X_copy = np.copy(X)
    Y_copy = np.copy(Y)
    if sampling != None:
        if sampling == "Random":
            X_copy, Y_copy = UnderSampling().fit_transform(X_copy, Y_copy)
        else:
            X_copy, Y_copy = OverSampling(sampling).fit_transform(X_copy, Y_copy)

    X_aux= np.copy(X_copy)

    if selector != None:
        instances, features = X_copy.shape
        n_features_to_keep = int(np.sqrt(features))
        fs = get_feature_selector(selector, n_features_to_keep, instances, features)
        X_aux = fs.fit_transform(X_aux, Y_copy)
        if selector == "PCA":


            indexes = fs.components_
            return indexes

    indexes = get_indexes(X_copy, X_aux)
    return indexes


def get_indexes(X1: np.ndarray , X2: np.ndarray ):
    list = []
    for i in range(X1.shape[1]):
        for j in range(X2.shape[1]):
            a = X1[:,i]
            b = X2[:,j]

            if np.array_equal (a,b):
                list.append(i)

    return list

if __name__ == '__main__':
    SCORING = ['accuracy', 'precision', 'recall', 'f1']
    main('f1', 10)
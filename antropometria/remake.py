import math
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from classifiers.KNearestNeighbors import KNearestNeighbors as Knn
from classifiers.NaiveBayes import NaiveBayes as Nb
from classifiers.NeuralNetwork import NeuralNetwork as Nn
from classifiers.RandomForests import RandomForests as Rf
from classifiers.SupportVectorMachine import SupportVectorMachine as Svm
from config import logger
from config.constants.training import SAMPLINGS, REDUCTIONS
from config.constants.general import CV, N_SPLITS
from feature_selectors.utils.getter import get_feature_selector
from sampling.OverSampling import OverSampling
from sampling.UnderSampling import UnderSampling
from utils.dataset.load import LoadData

log = logger.get_logger(__file__)
CLASSIFIERS = [Nb ]#Svm,, Nn , Rf,Knn


def main(metrica: str):
    dataset, labels = LoadData('dlibHOG', 'distances_all_px_eu', ["casos", "controles"]).load()
    numpy_dataset = dataset.to_numpy()
    nfeatures = int(math.sqrt(numpy_dataset.shape[1]))

    all_x = []
    all_y = []
    for train_index, test_index in CV.split(numpy_dataset, labels):
        x_aux, y_aux = numpy_dataset[train_index], labels[train_index]
        x_test, y_test = numpy_dataset[test_index], labels[test_index]

        x_train, x_calibration, y_train, y_calibration = train_test_split(x_aux, y_aux, stratify=y_aux, train_size=0.89)

        all_x.append([x_train, x_test, x_calibration])
        all_y.append([y_train, y_test, y_calibration])

    number_of_features_set = []
    for balanceador in SAMPLINGS:
        for redutor in REDUCTIONS:
            for x, y in zip(all_x, all_y):
                x_training = x[0]
                y_training = y[0]

                number_of_features_set.append(get_best_features(x_training, y_training, balanceador, redutor))
    non_redundant_feature_sets = best_reductions(number_of_features_set, nfeatures)

    for indutor in CLASSIFIERS:
        for balanceador in SAMPLINGS:
            for n in range(len(non_redundant_feature_sets)):

                reductions = non_redundant_feature_sets[n]
                media_ideal = 0.0
                model = indutor(n_features=len(reductions))

                for parametros in ParameterGrid(model.parameter_grid):
                    estimator = model.estimator
                    estimator.set_params(**parametros)
                    soma = 0.0
                    for x, y in zip(all_x, all_y):
                        x_train, x_test, x_calibration = x
                        y_train, y_test, y_calibration = y

                        if balanceador is not None:
                            if balanceador == "Random":
                                x_train, y_train = UnderSampling().fit_transform(x_train, y_train)
                            else:
                                x_train, y_train = OverSampling(balanceador).fit_transform(x_train, y_train)
                        x_train = apply_reductions(x_train, reductions)
                        x_calibration = apply_reductions(x_calibration, reductions)

                        estimator.fit(x_train, y_train)
                        y_predict = estimator.predict(x_calibration)

                        if metrica == "f1":
                            f1 = f1_score(y_calibration, y_predict)
                            soma += f1
                        elif metrica == "accuracy":
                            acuracia = accuracy_score(y_calibration, y_predict)
                            soma += acuracia
                        elif metrica == 'precision':
                            precisao = precision_score(y_calibration, y_predict)
                            soma += precisao
                        else:
                            recall = recall_score(y_calibration, y_predict)
                            soma += recall

                    media = soma / N_SPLITS
                    print(media)
                    print("+++++++++")
                    if media > media_ideal:
                        media_ideal = media
                    parametros_otimos = parametros
                print(parametros_otimos)
                estimator_final = model.estimator
                print(media_ideal)
                print("*******************")
                media_otima, soma = 0, 0
                estimator_final.set_params(**parametros_otimos)
                for x, y in zip(all_x, all_y):

                    x_train, x_test, x_calibration = x
                    y_train, y_test, y_calibration = y

                    x_train = np.concatenate((x_train, x_calibration), axis=0)
                    y_train = np.concatenate((y_train, y_calibration), axis=0)

                    if balanceador == "Random":
                        x_train, y_train = UnderSampling().fit_transform(x_train, y_train)
                    else:
                        x_train, y_train = OverSampling(balanceador).fit_transform(x_train, y_train)

                    apply_reductions(x_train, non_redundant_feature_sets[n])
                    apply_reductions(x_test, non_redundant_feature_sets[n])

                    estimator_final.fit(x_train, y_train)
                    y_test_predict = estimator_final.predict(x_test)
                    if metrica == "f1":
                        f1 = f1_score(y_test, y_test_predict)
                        soma += f1
                    elif metrica == "accuracy":
                        acuracia = accuracy_score(y_test, y_test_predict)
                        soma += acuracia
                    elif metrica == 'precision':
                        precisao = precision_score(y_test, y_test_predict)
                        soma += precisao
                    else:
                        recall = recall_score(y_test, y_test_predict)
                        soma += recall
                media_otima = soma / N_SPLITS
                print(media_otima)


def apply_reductions(x: np.ndarray, characteristics):
    deletar = []
    for i in range(x.shape[1]):
        if i not in characteristics:
            deletar.append(i)
    return np.delete(x, deletar, axis=1)


def best_reductions(characteristics, nfeatures):
    non_redundant_features_sets = []
    for i in characteristics:
        if i not in non_redundant_features_sets:
            non_redundant_features_sets.append(i)

    numpyarr = np.array(characteristics)
    flat = numpyarr.flatten()
    consensus_features_np = np.bincount(flat)[0:nfeatures]
    consensus_features = consensus_features_np.tolist()
    if consensus_features not in non_redundant_features_sets:
        non_redundant_features_sets.append(consensus_features)

    return non_redundant_features_sets


def get_best_features(x_training: np.ndarray, y_training: np.ndarray, sampling: str = None, selector: str = None):
    x, y = np.copy(x_training), np.copy(y_training)
    if sampling is not None:
        if sampling == "Random":
            x, y = UnderSampling().fit_transform(x, y)
        else:
            x, y = OverSampling(sampling).fit_transform(x, y)
    x_without_reductions = np.copy(x)
    if selector is not None:
        instances, features = x.shape
        n_features_to_keep = int(np.sqrt(features))
        feature_selector = get_feature_selector(selector, n_features_to_keep, instances, features)
        x = feature_selector.fit_transform(x, y)



    return get_indexes(x_without_reductions, x, selector)


def get_indexes(original_dataset: np.ndarray, reduced_dataset: np.ndarray, selector: str ):
    if selector is None:
        return [i for i in range(original_dataset.shape[1])]
    indexes = []
    for i in range(original_dataset.shape[1]):
        for j in range(reduced_dataset.shape[1]):
            if np.array_equal (original_dataset[:,i],reduced_dataset[:,j]):
                indexes.append(i)

    return indexes

if __name__ == '__main__':
    main('f1')

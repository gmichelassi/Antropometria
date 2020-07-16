# Classifiers
from classifiers import svm, rf, knn, nnn, nb
from mainDimensionalityReduction import run_dimensionality_reductions

# Classifiers evaluation methods
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Utils
from multiprocessing import Process
import matplotlib as plt
from classifiers.utils import mean_scores, sample_std
import csv
import numpy as np
import pandas as pd
import time
import math
import initContext as context
from config import logger

plt.set_loglevel("info")
context.loadModules()
log = logger.getLogger(__file__)


def __completeFrame(X, y, synthetic_X, synthetic_y, n_splits=10, current_fold=0):
    synthetic_X = np.array_split(synthetic_X, n_splits)
    synthetic_y = np.array_split(synthetic_y, n_splits)

    for i in range(len(synthetic_X)):
        if i == current_fold:
            pass
        else:
            for j in range(len(synthetic_X[i])):
                X = np.append(X, [synthetic_X[i][j]], 0)
                y = np.append(y, [synthetic_y[i][j]], 0)

    return X, y


def __errorEstimation(lib='dlibHOG', dataset='', model=None, parameters=None, reduction='None', filtro=0.0, amostragem=None, min_max=False):
    if parameters is None or model is None:
        return

    log.info("Running error estimation for current classifier with best parameters found")

    try:
        X, y, synthetic_X, synthetic_y = run_dimensionality_reductions(lib=lib, dataset=dataset, reduction=reduction, filtro=filtro, amostragem=amostragem, split_synthetic=True, min_max=min_max, verbose=False)
    except RuntimeError as re:
        log.info("It was not possible run error estimation for this test")
        log.info("Error: " + str(re))
        return {'accuracy': "", 'IC': "", 'precision': "", 'recall': "", 'f1score': "", 'time': ""}

    model = model.set_parameters(parameters)

    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits)

    current_fold, folds = 0, []
    for train_index, test_index in cv.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        if synthetic_X is not None:
            X_train, y_train = __completeFrame(X_train, y_train, synthetic_X, synthetic_y, n_splits,
                                               current_fold)
        folds.append((X_train, y_train, X_test, y_test))
        current_fold += 1

    accuracy, precision, recall, f1, tempo = [], [], [], [], []

    # Antes, somente geramos as folds, agora vamos utilizadas para cada modelo
    for i in range(n_splits):
        __start_time = time.time()
        model.fit(folds[i][0], folds[i][1])
        y_predict = model.predict(folds[i][2])

        accuracy.append(accuracy_score(y_true=folds[i][3], y_pred=y_predict))
        precision.append(precision_score(y_true=folds[i][3], y_pred=y_predict))
        recall.append(recall_score(y_true=folds[i][3], y_pred=y_predict))
        f1.append(f1_score(y_true=folds[i][3], y_pred=y_predict))
        tempo.append((time.time() - __start_time) / 60)

    mean_results = mean_scores({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'time': tempo})

    tc = 2.262  # valor na tabela da distribuição t-student com k-1 graus de liberdade e p = ?
    s = sample_std(accuracy)
    IC_2 = mean_results['accuracy'] + tc * (s / math.sqrt(n_splits))
    IC_1 = mean_results['accuracy'] - tc * (s / math.sqrt(n_splits))

    IC = (IC_1, IC_2)

    log.info("Accuracy found: {0}".format(mean_results['accuracy']))
    log.info("Confidence interval: {0}".format(IC))
    log.info("Precision found: {0}".format(mean_results['precision']))
    log.info("Recall found: {0}".format(mean_results['recall']))
    log.info("F1 Score found: {0}".format(mean_results['f1']))

    return {
        'accuracy': mean_results['accuracy'],
        'IC': IC,
        'precision': mean_results['precision'],
        'recall': mean_results['recall'],
        'f1score': mean_results['f1'],
        'time': mean_results['time']
    }


def run_gridSearch(lib='dlibHOG', dataset='euclidian_px_all'):
    log.info("Running Grid Search for %s dataset", dataset)

    isRandomForestDone = False
    dimensionality_reductions = ['None', 'PCA', 'mRMR', 'FCBF', 'CFS', 'RFS', 'ReliefF']
    classifiers = {'randomforestclassifier': rf, 'svc': svm, 'kneighborsclassifier': knn, 'mlpclassifier': nnn, 'gaussiannb': nb}
    amostragens = [None, 'Smote']  # 'Random', 'Borderline', 'KMeans', 'SVM', 'Tomek'
    filtros = [0.0]  # , 0.98, 0.99
    min_maxs = [False]  # , True

    for classifier in classifiers.keys():
        for reduction in dimensionality_reductions:

            if isRandomForestDone and classifier == 'randomforestclassifier':
                continue
            elif not isRandomForestDone and classifier == 'randomforestclassifier':
                isRandomForestDone = True
            elif classifier != 'randomforestclassifier' and reduction == 'None':
                continue

            for filtro in filtros:
                for min_max in min_maxs:
                    for amostragem in amostragens:
                        start_processing = time.time()

                        log.info("Running test for [lib: %s, classifier: %s, reduction: %s, filter: %s, min_max: %s, sampling: %s]", lib, classifier, reduction, filtro, min_max, amostragem)

                        try:
                            X, y, synthetic_X, synthetic_y = run_dimensionality_reductions(lib=lib, dataset=dataset, reduction=reduction, filtro=filtro, amostragem=amostragem, split_synthetic=False, min_max=min_max)
                        except RuntimeError as re:
                            log.info("It was not possible run test for [classifier: %s, reduction: %s, filter: %s, min_max: %s, sampling: %s]", classifier, reduction, filtro, min_max, amostragem)
                            log.info("Error: " + str(re))
                            continue

                        instances, features = X.shape
                        if classifier == 'randomforestclassifier':
                            n_features_to_keep = int(np.sqrt(features))
                        else:
                            n_features_to_keep = features

                        n_splits = 10
                        scoring = {'accuracy': 'accuracy', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro', 'f1_macro': 'f1_macro'}
                        estimator, param_grid = classifiers[classifier].make_grid_optimization_pipes(n_features_to_keep)
                        cv = StratifiedKFold(n_splits=n_splits)

                        try:
                            log.info("Training Models for %s and %s", classifier, reduction)

                            grd = GridSearchCV(
                                estimator=estimator,
                                param_grid=param_grid,
                                scoring=scoring,
                                cv=cv,
                                refit='accuracy',
                                return_train_score=False,
                                n_jobs=-1
                                # -1 means all CPUs
                                # For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
                            )

                            grid_results = grd.fit(X, y)

                            log.info("Training complete")

                        except ValueError as e:
                            log.exception("Exception during pipeline execution", extra=e)
                            grid_results = None
                        except KeyError as ke:
                            log.exception("Exception during pipeline execution", extra=ke)
                            grid_results = None

                        if grid_results is not None:
                            log.info("Best result presented accuracy %.2f%% for test [classifier: %s, reduction: %s, filter: %s, min_max: %s, sampling: %s]", grid_results.best_score_ * 100, classifier, reduction, filtro, min_max, amostragem)
                            log.info("Best parameters found: {0}".format(grid_results.best_params_))
                            log.info("Best parameters were found on index: {0}".format(grid_results.best_index_))

                            try:
                                errResults = __errorEstimation(lib=lib, dataset=dataset, model=classifiers[classifier], parameters=grid_results.best_params_, reduction=reduction, filtro=filtro, amostragem=amostragem, min_max=min_max)
                            except KeyError as ke:
                                errResults = {'accuracy': '', 'IC': '', 'precision': '', 'recall': '', 'f1score': '', 'time': ''}
                                log.info("It was not possible to run error estimation")
                                log.info("Error: {0}".format(ke))

                            log.info("Saving results!")
                            with open(f"./output/GridSearch/{lib}_best_results.csv", "a") as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=['biblioteca', 'classifier', 'reduction', 'filtro', 'min_max', 'par_amostragem', 'par_accuracy', 'err_accuracy', 'IC', 'err_precision', 'err_recall', 'err_f1score', 'err_time', 'parameters'])
                                results = {
                                    'biblioteca': lib,
                                    'classifier': classifier,
                                    'reduction': reduction,
                                    'filtro': filtro,
                                    'min_max': min_max,
                                    'par_amostragem': amostragem,
                                    'par_accuracy': grid_results.best_score_,
                                    'err_accuracy': errResults['accuracy'],
                                    'IC': errResults['IC'],
                                    'err_precision': errResults['precision'],
                                    'err_recall': errResults['recall'],
                                    'err_f1score': errResults['f1score'],
                                    'err_time': errResults['time'],
                                    'parameters': grid_results.best_params_}
                                writer.writerow(results)

                            df_results = pd.DataFrame(grid_results.cv_results_)
                            df_results.drop('params', axis=1)
                            path_results = './output/GridSearch/results_{0}_{1}_{2}_{3}_{4}_{5}_{6}.csv'.format(lib, dataset, classifier, reduction, filtro, min_max, amostragem)
                            df_results.to_csv(path_results, index_label='id')

                        log.info("Execution time: %s minutes" % ((time.time() - start_processing) / 60))


def run_randomizedSearch(dataset='euclidian_px_all', filtro=0.0):
    log.info("Running Randomized Search for %s dataset", dataset)

    dimensionality_reductions = ['None', 'PCA', 'mRMRProxy', 'FCBFProxy',
                                 'CFSProxy', 'RFSProxy', 'ReliefF']
    reduction = dimensionality_reductions[0]

    classifiers = {'randomforestclassifier': rf,
                   'svc': svm,
                   'kneighborsclassifier': knn,
                   'mlpclassifier': nnn
                   }

    for classifier in classifiers.keys():
        samples, labels = run_dimensionality_reductions(filtro=filtro, reduction=reduction)

        instances, features = samples.shape
        n_features_to_keep = int(np.sqrt(features))

        scoring = {'accuracy': 'accuracy',
                   'precision_macro': 'precision_macro',
                   'recall_macro': 'recall_macro',
                   'f1_macro': 'f1_macro'}

        estimators, param_distributions, classifier_name = \
            classifiers[classifier].make_random_optimization_pipes(n_features_to_keep)

        cv = StratifiedKFold(n_splits=4)
        for estimator in estimators:
            try:
                log.info("Training Models for %s and %s", classifier_name, reduction)

                rdm = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=param_distributions,
                    n_iter=100,
                    scoring=scoring,
                    n_jobs=-1,
                    iid=False,
                    cv=cv,
                    refit='accuracy',
                    random_state=787870)
                rdm_results = rdm.fit(samples, labels)

                log.info("Training complete")

            except ValueError as e:
                log.exception("Exception during pipeline execution", extra=e)
                rdm_results = None
            except KeyError as ke:
                log.exception("Exception during pipeline execution", extra=ke)
                rdm_results = None

            if rdm_results is not None:
                log.info("Best result presented accuracy %.2f%% for %s and %s",
                         rdm_results.best_score_ * 100, classifier_name, reduction)
                log.info("Best parameters found: " + str(rdm_results.best_params_))
                log.info("Best parameters were found on index: " + str(rdm_results.best_index_))
                log.info("Saving results!")
                df_results = pd.DataFrame(rdm_results.cv_results_)
                df_results.drop('params', axis=1)
                path_results = './output/RandomSearch/results_' \
                               + dataset + '_' + classifier_name + '_' + reduction + '.csv'
                df_results.to_csv(path_results, index_label='id')


if __name__ == '__main__':
    start_time = time.time()
    # run_gridSearch('dlibCNN', 'euclidian_px_all')  # dostoievski
    # run_gridSearch('openFace', 'euclidian_px_all')  # tolstoi
    # run_gridSearch('openCvDNN', 'euclidian_px_all')  # puchkin
    run_gridSearch('panda', 'olhaPanda')
    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

# Classifiers
from classifiers import svm, rf, knn, nnn, nb
from mainDimensionalityReduction import run_dimensionality_reductions

# Classifiers evaluation methods
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Utils
from classifiers.utils import mean_scores, sample_std
import csv
import numpy as np
import pandas as pd
import time
import math
import initContext as context
from config import logger

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


def run_gridSearch(dataset='euclidian_px_all', filtro=0.0, amostragem=None, min_max=False):
    log.info("Running Grid Search for %s dataset", dataset)

    isRandomForestDone = False
    dimensionality_reductions = ['None', 'PCA', 'mRMR', 'FCBF', 'CFS', 'RFS', 'ReliefF']
    classifiers = {'randomforestclassifier': rf,
                   'svc': svm,
                   'kneighborsclassifier': knn,
                   'mlpclassifier': nnn,
                   'gaussiannb': nb
                   }

    for classifier in classifiers.keys():
        for reduction in dimensionality_reductions:
            if isRandomForestDone and classifier == 'randomforestclassifier':
                continue
            elif not isRandomForestDone and classifier == 'randomforestclassifier':
                isRandomForestDone = True
            elif classifier != 'randomforestclassifier' and reduction == 'None':
                continue

            X, y, synthetic_X, synthetic_y = run_dimensionality_reductions(reduction, filtro, amostragem)

            instances, features = X.shape
            if classifier == 'randomforestclassifier':
                n_features_to_keep = int(np.sqrt(features))
            else:
                n_features_to_keep = features

            estimators = classifiers[classifier].make_grid_optimization_estimators(n_features_to_keep)
            n_splits = 10
            cv = StratifiedKFold(n_splits=n_splits)

            test_id, best_accuracy, best_id, best_parameters, best_IC = 0, -1, -1, {}, ()

            columns = ['id', 'dataset', 'classifier', 'reduction', 'accuracy', 'IC_Accuracy', 'precision', 'recall', 'f1']
            columns += classifiers[classifier].getParams()

            with open('./output/GridSearch/results_{0}_{1}.csv'.format(classifier, reduction), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()

                log.info("Training Models for %s and %s", classifier, reduction)

                for estimator in estimators:
                    log.info("#%d - Executing Cross-Validation", test_id)

                    accuracy, precision, recall, f1 = [], [], [], []
                    current_fold = 0
                    for train_index, test_index in cv.split(X, y):
                        X_train, y_train = X[train_index], y[train_index]
                        X_test, y_test = X[test_index], y[test_index]

                        if synthetic_X is not None:
                            X_train, y_train = __completeFrame(X_train, y_train, synthetic_X, synthetic_y, n_splits, current_fold)

                        estimator.fit(X_train, y_train)
                        y_predict = estimator.predict(X_test)

                        accuracy.append(accuracy_score(y_true=y_test, y_pred=y_predict))
                        precision.append(precision_score(y_true=y_test, y_pred=y_predict))
                        recall.append(recall_score(y_true=y_test, y_pred=y_predict))
                        f1.append(f1_score(y_true=y_test, y_pred=y_predict))
                        # c_matrix = confusion_matrix(y_true=y_test, y_pred=y_predict)
                        current_fold += 1

                    log.info("#%d - Cross-Validation success!", test_id)

                    parameters = estimator.get_params(deep=False)
                    mean_results = mean_scores({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

                    tc = 2.262  # valor na tabela da distribuição t-student com k-1 graus de liberdade e p = ?
                    s = sample_std(accuracy)
                    IC_1 = mean_results['accuracy'] + tc * (s / math.sqrt(n_splits))
                    IC_2 = mean_results['accuracy'] - tc * (s / math.sqrt(n_splits))

                    IC = (IC_2, IC_1)

                    if mean_results['accuracy'] > best_accuracy:
                        best_accuracy = mean_results['accuracy']
                        best_id = test_id
                        best_parameters = parameters
                        best_IC = IC

                    log.info("#%d - CV result (accuracy) %.2f for model %s and reduction %s",
                             test_id, mean_results['accuracy'], classifier, reduction)

                    results = {'id': test_id,
                               'dataset': dataset,
                               'classifier': classifier,
                               'reduction': reduction,
                               'accuracy': mean_results['accuracy'],
                               'IC_Accuracy': IC,
                               'precision': mean_results['precision'],
                               'recall': mean_results['recall'],
                               'f1': mean_results['f1']}

                    results.update(parameters)
                    writer.writerow(results)
                    log.info("#%d - Saving results!", test_id)

                    test_id += 1
                    p_done = (100 * float(test_id)) / float(len(estimators))
                    log.info("%.2f%% of classifier %s processing done...", p_done, classifier)

                log.info("Best result presented accuracy %.3f for %s and %s", best_accuracy, classifier, reduction)
                log.info("Confidence interval is [%.3f,%.3f]", best_IC[0], best_IC[1])
                log.info("Best parameters found: {0}".format(best_parameters))
                log.info("Best parameters were found on index: {0}".format(best_id))


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
    run_gridSearch()
    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

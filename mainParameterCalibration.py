# Classifiers
from classifiers import svm, rf, knn, nnn, nb
from mainDimensionalityReduction import run_dimensionality_reductions

# Classifiers evaluation methods
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

# Utils
import numpy as np
import pandas as pd
import time
import initContext as context
from config import logger

context.loadModules()
log = logger.getLogger(__file__)


def run_gridSearch(dataset='euclidian_px_all', filtro=0.0, amostragem=None, min_max=False):
    isRandomForestDone = False
    log.info("Running Grid Search for %s dataset", dataset)

    dimensionality_reductions = ['None', 'PCA', 'mRMR', 'FCBF',
                                 'CFS', 'RFS', 'ReliefF']

    classifiers = {'randomforestclassifier': rf,
                   'svc': svm,
                   'kneighborsclassifier': knn,
                   'mlpclassifier': nnn
                   }

    for classifier in classifiers.keys():
        for reduction in dimensionality_reductions:
            if isRandomForestDone and classifier == 'randomforestclassifier':
                continue
            elif not isRandomForestDone and classifier == 'randomforestclassifier':
                isRandomForestDone = True
            elif classifier != 'randomforestclassifier' and reduction == 'None':
                continue

            samples, labels = run_dimensionality_reductions(filtro=filtro, reduction=reduction,
                                                            amostragem=amostragem, min_max=min_max)

            instances, features = samples.shape
            n_features_to_keep = int(np.sqrt(features))

            scoring = {'accuracy': 'accuracy',
                       'precision_macro': 'precision_macro',
                       'recall_macro': 'recall_macro',
                       'f1_macro': 'f1_macro'}

            estimators, param_grid, classifier_name = classifiers[classifier].make_grid_optimization_pipes(
                n_features_to_keep)
            cv = StratifiedKFold(n_splits=4)
            for estimator in estimators:
                try:
                    log.info("Training Models for %s and %s", classifier_name, reduction)

                    grd = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=cv,
                        refit='accuracy',
                        return_train_score=False,
                        n_jobs=-1  # -1 means all CPUs
                        # For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
                    )
                    grid_results = grd.fit(samples, labels)

                    log.info("Training complete")

                except ValueError as e:
                    log.exception("Exception during pipeline execution", extra=e)
                    grid_results = None
                except KeyError as ke:
                    log.exception("Exception during pipeline execution", extra=ke)
                    grid_results = None

                if grid_results is not None:
                    log.info("Best result presented accuracy %.2f%% for %s and %s",
                             grid_results.best_score_ * 100, classifier_name, reduction)
                    log.info("Best parameters found: {0}".format(grid_results.best_params_))
                    log.info("Best parameters were found on index: {0}".format(grid_results.best_index_))

                    log.info("Saving results!")
                    df_results = pd.DataFrame(grid_results.cv_results_)
                    df_results.drop('params', axis=1)
                    path_results = './output/GridSearch/results_{0}_{1}_{2}.csv'.format(dataset,
                                                                                        classifier_name,
                                                                                        reduction)
                    df_results.to_csv(path_results, index_label='id')


def run_randomizedSearch(dataset='euclidian_px_all', filtro=0.0):
    log.info("Running Randomized Search for %s dataset", dataset)

    dimensionality_reductions = ['None', 'PCA', 'mRMRProxy', 'FCBFProxy',
                                 'CFSProxy', 'RFSProxy', 'ReliefF']
    reduction = dimensionality_reductions[0]

    classifiers = {'randomforestclassifier': rf,
                   # 'svc': svm,
                   # 'kneighborsclassifier': knn,
                   # 'mlpclassifier': nnn
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

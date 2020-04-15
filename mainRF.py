# Classifiers
from sklearn.ensemble import RandomForestClassifier
from classifiers import rf

# Classifiers evaluation methods
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix

# DataSets
import asd_data as asd

# Utils
import pandas as pd
import numpy as np
import csv
from sklearn.utils.multiclass import unique_labels
import time
from classifiers.utils import mean_scores, save_confusion_matrix, save_cross_val_scores
from classifiers.utils import calculate_metrics, apply_pearson_feature_selection

from config import logger
import initContext as context

context.loadModules()
log = logger.getLogger(__file__)


def run_combinations(filtro=0, verbose=False):
    all_samples = {}

    X, y = asd.load_data(d_type='euclidian', unit='px', m='', normalization='', dataset='all', labels=False)
    all_samples['euclidian_px_all'] = (X, y)

    for k in all_samples.keys():
        log.info("Running models for " + k + " dataset")
        samples, labels = all_samples[k]
        log.info("X.shape " + str(samples.shape) + ", y.shape " + str(labels.shape))
        if filtro != 0:
            samples = apply_pearson_feature_selection(samples, filtro)
        else:
            samples = samples.values
        n_classes = len(unique_labels(labels))

        n_samples = samples.shape[0]
        n_features = samples.shape[1]

        instances, features = samples.shape
        log.info('Data has {0} classes, {1} instances and {2} features'.format(n_classes, instances, features))

        n_features_to_keep = int(np.sqrt(features))

        dimensionality_reductions = None

        pipes, reductions_names, models_names = [], [], []
        for m in [rf]:
            pipe, reductions_name, models_name = m.make_pipes(dimensionality_reductions, n_features_to_keep)
            pipes += pipe
            reductions_names += reductions_name
            models_names += models_name

        log.info('Total de modelos {0}'.format(len(pipes)))

        columns = [
            'id', 'precision', 'recall', 'f1', 'accuracy', 'dimensionality_reduction', 'error', 'classifier',
            'dataset', 'n_classes']

        classifiers = [RandomForestClassifier()]
        for classifier in classifiers:
            columns += classifier.get_params().keys()

        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'}

        with open('./output/bloco-rf-' + k + '-2019.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            id = 0

            for current_pipe, reduction, model_name in zip(pipes, reductions_names, models_names):
                try:
                    log.info("#%d - Executing Cross-Validation", id)
                    cv = StratifiedKFold(n_splits=4)
                    cv_results = cross_validate(
                        estimator=current_pipe,
                        X=samples,
                        y=labels,
                        scoring=scoring,
                        cv=cv,
                        n_jobs=-1)  # -1 means all CPUs
                    # For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.

                    if verbose:
                        cm = 0
                        CMatrix = {}
                        for train_index, val_index in cv.split(X=samples, y=labels):
                            current_pipe.fit(samples[train_index], labels[train_index])
                            y_pred = current_pipe.predict(samples[val_index])
                            CMatrix['CV-' + str(cm)] = pd.DataFrame(
                                confusion_matrix(labels[val_index], y_pred),
                                index=[i for i in unique_labels(labels)],
                                columns=[i for i in unique_labels(labels)])
                            cm += 1
                    log.info("#%d - Cross-Validation success!", id)

                except ValueError as e:
                    log.exception("Exception during pipeline execution", extra=e)
                    cv_results = None
                except KeyError as ke:
                    log.exception("Exception during pipeline execution", extra=ke)
                    cv_results = None

                if cv_results is not None:
                    if verbose:
                        calculate_metrics(CMatrix, labels)
                        save_cross_val_scores(id, k,
                                              cv_results['test_precision_macro'],
                                              cv_results['test_recall_macro'],
                                              cv_results['test_f1_macro'],
                                              model_name, reduction)
                        save_confusion_matrix(CMatrix, k, reduction, model_name, id)

                    mean_cv_results = mean_scores(cv_results)
                    log.info("#%d - CV result (accuracy) %.2f for model %s and reduction %s",
                             id, mean_cv_results['test_accuracy'], model_name, reduction)

                    results = {
                        'id': id,
                        'precision': mean_cv_results['test_precision_macro'],
                        'recall': mean_cv_results['test_recall_macro'],
                        'f1': mean_cv_results['test_f1_macro'],
                        'accuracy': mean_cv_results['test_accuracy'],
                        'error': 1 - mean_cv_results['test_accuracy'],
                        'dimensionality_reduction': reduction,
                        'classifier': model_name,
                        'dataset': k}

                    model = current_pipe.named_steps[model_name]
                    params = model.get_params(deep=False)
                    log.info("#%d - Saving results!", id)
                    results.update(params)
                    writer.writerow(results)
                id += 1
                p_done = (100 * float(id)) / float(len(pipes))
                log.info("%.3f %% of dataset %s processing done...", p_done, k)


if __name__ == '__main__':
    start_time = time.time()
    run_combinations()
    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

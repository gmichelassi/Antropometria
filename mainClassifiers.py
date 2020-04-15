# Classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from classifiers import svm, nb, nnn, knn

# Classifiers evaluation methods
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix

# Utils
import numpy as np
import pandas as pd
import csv
from sklearn.utils.multiclass import unique_labels
from classifiers.utils import mean_scores, save_confusion_matrix, save_cross_val_scores
from classifiers.utils import calculate_metrics
import time
import initContext as context
from config import logger

context.loadModules()
log = logger.getLogger(__file__)


def run_combinations(dataset='euclidian_px_all', verbose=False):
    """
    This function loads the datasets and labels pre-processed by Dimensionality Reductions and trains four classifiers
    with a set of pre-defined parameters.

    Input
    ------
    dataset: {string} The name of the dataset file that will be loaded
    verbose: {boolean} If true, extra information will be saved
                            - Confusion Matrix, metrics for each cross validation...

    Output
    ------
    CSV file with the evaluation metrics for each classifier that was trained
    P.S.: The CSV file that will be saved will be grouped by dimensionality reduction method applied
    """
    log.info("Running models for %s dataset", dataset)

    dimensionality_reductions = ['PCA',
                                 'mRMRProxy',
                                 'FCBFProxy',
                                 'CFSProxy',
                                 'RFSProxy',
                                 'ReliefF'
                                 ]

    labels = pd.read_csv(filepath_or_buffer='./data/reduction_files/labels_distances_euclidian_px_all.csv')
    labels = np.array(labels).reshape(111,)  # Convert a pandas dataframe to a 1D numpy array

    n_classes = unique_labels(labels)
    n_features_to_keep = 47

    for dimensionality_reduction in dimensionality_reductions:
        path = './data/reduction_files/{0}_distances_{1}.csv'.format(dimensionality_reduction, dataset)

        samples = pd.read_csv(filepath_or_buffer=path)
        samples = samples.to_numpy()
        log.info("X.shape %s, y.shape %s", str(samples.shape), str(labels.shape))

        pipes, models_names = [], []
        for m in [svm, nb, knn, nnn]:
            pipe, models_name = m.make_pipes()
            pipes += pipe
            models_names += models_name

        log.info('Total de modelos {0}'.format(len(pipes)))

        columns = [
            'id', 'precision', 'recall', 'f1', 'accuracy', 'dimensionality_reduction', 'error', 'classifier',
            'dataset', 'n_classes', 'n_features']

        classifiers = [SVC(), GaussianNB(), KNeighborsClassifier(), MLPClassifier()]
        for classifier in classifiers:
            columns += classifier.get_params().keys()

        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'}

        with open('./output/bloco-svm-nb-knn-nn-{0}-2019.csv'.format(dimensionality_reduction), 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            id = 0
            for current_pipe, model_name in zip(pipes, models_names):
                try:
                    log.info("#%d - Executing Cross-Validation", id)
                    cv = StratifiedKFold(n_splits=4)
                    cv_results = cross_validate(
                        estimator=current_pipe,
                        X=samples,
                        y=labels,
                        scoring=scoring,
                        cv=cv,
                        n_jobs=-1)  # all CPUs

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
                    # Se verbose == true, exibimos/salvamos algumas informações extras sobre a classificação
                    # Se não, é salvo somente as informações Default
                    if verbose:
                        calculate_metrics(CMatrix, labels)
                        save_cross_val_scores(id, dataset,
                                              cv_results['test_precision_macro'],
                                              cv_results['test_recall_macro'],
                                              cv_results['test_f1_macro'],
                                              model_name, dimensionality_reduction)
                        save_confusion_matrix(CMatrix, dataset, dimensionality_reduction, model_name, id)

                    mean_cv_results = mean_scores(cv_results)
                    log.info("#%d - CV result (accuracy) %.2f for model %s and reduction %s",
                             id, mean_cv_results['test_accuracy'], model_name, dimensionality_reduction)

                    results = {
                        'id': id,
                        'precision': mean_cv_results['test_precision_macro'],
                        'recall': mean_cv_results['test_recall_macro'],
                        'f1': mean_cv_results['test_f1_macro'],
                        'accuracy': mean_cv_results['test_accuracy'],
                        'error': 1 - mean_cv_results['test_accuracy'],
                        'dimensionality_reduction': dimensionality_reduction,
                        'classifier': model_name,
                        'n_features': n_features_to_keep,
                        'n_classes': n_classes,
                        'dataset': dataset}

                    model = current_pipe.named_steps[model_name]
                    params = model.get_params(deep=False)
                    log.info("#%d - Saving results!", id)
                    results.update(params)
                    writer.writerow(results)
                id += 1
                p_done = (100 * float(id)) / float(len(pipes))
                log.info("%.3f %% of dataset %s processing done...", p_done, dataset)


if __name__ == '__main__':
    start_time = time.time()
    run_combinations()
    log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

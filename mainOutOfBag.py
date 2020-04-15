# Classifiers
from sklearn.ensemble import RandomForestClassifier
from classifiers import rf

# DataSets
import asd_data as asd

# Utils
import numpy as np
import csv
from sklearn.utils.multiclass import unique_labels
import time
from classifiers.utils import apply_pearson_feature_selection

from config import logger
import initContext as context
context.loadModules()
log = logger.getLogger(__file__)


def run_combinations():
    """
    This functions evaluate the out-of-bag score from the random forest classifier training models with different sets
    of parameters

    Output
    ------
    CSV file with the out-of-bag score for each set of random forests parameters
    """
    all_samples = {}

    X, y = asd.load_data(d_type='euclidian', unit='px', m='', dataset='all', labels=False)
    all_samples['euclidian_px_all'] = (X, y)

    for k in all_samples.keys():
        log.info("Running models for " + k + " dataset")
        samples, labels = all_samples[k]
        log.info("X.shape " + str(samples.shape) + ", y.shape " + str(labels.shape))
        # samples = samples.values
        samples = apply_pearson_feature_selection(samples, 0.98)
        n_classes = len(unique_labels(labels))

        instances, features = samples.shape
        log.info('Data has {0} classes, {1} instances and {2} features'.format(n_classes, instances, features))

        n_features_to_keep = int(np.sqrt(features))

        dimensionality_reductions = None

        pipes, reductions_names, models_names = [], [], []
        for m in [rf]:
            pipe, reductions_name, models_name = m.make_oob_pipes(dimensionality_reductions, n_features_to_keep)
            pipes += pipe
            reductions_names += reductions_name
            models_names += models_name

        log.info('Total de modelos {0}'.format(len(pipes)))

        columns = [
            'id', 'classifier', 'out-of-bag', 'dataset', 'n_classes', 'n_features']

        classifiers = [RandomForestClassifier()]
        for classifier in classifiers:
            columns += classifier.get_params().keys()

        with open('./output/rf-out-of-bag-' + k + '-.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            id = 0

            for current_pipe, reduction, model_name in zip(pipes, reductions_names, models_names):
                try:
                    current_pipe.fit(samples, labels)
                    oob_score_ = current_pipe.oob_score_
                except ValueError as e:
                    log.exception("Exception during pipeline execution", extra=e)
                    oob_score_ = None
                except KeyError as ke:
                    log.exception("Exception during pipeline execution", extra=ke)
                    oob_score_ = None

                if oob_score_ is not None:
                    log.info("#%d - Out of Bag score %.2f for model %s and reduction %s",
                             id, oob_score_, model_name, reduction)

                    results = {
                        'id': id,
                        'out-of-bag': oob_score_,
                        'classifier': model_name,
                        'dataset': k,
                        'n_classes': n_classes,
                        'n_features': n_features_to_keep}

                    model = current_pipe
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

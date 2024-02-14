import time

from antropometria.config import CV, SCORING
from antropometria.config import logger
from antropometria.utils.results import extract_results
from sklearn.model_selection import GridSearchCV


log = logger.get_logger(__file__)


def grid_search(classifier, x, y):
    log.info(f'Running cross validation for {classifier.__name__}')

    initial_time = time.time()
    n_instances, n_features = x.shape
    model = classifier(n_features=n_features)
    
    grd = GridSearchCV(
        estimator=model.estimator,
        param_grid=model.parameter_grid,
        scoring=SCORING,
        cv=CV,
        refit='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_results = grd.fit(x, y)

    accuracy, precision, recall, f1, parameters, best_estimator = extract_results(grid_results)
    ellapsed_time = (time.time() - initial_time) / 60

    log.info(f"Finished current grid search in {ellapsed_time:.2f} minutes")
    log.info(f'Results presented f1-score {(f1 * 100):.2f}%.')
    log.info(f'Best parameters found: {parameters}')

    return accuracy, precision, recall, f1, parameters, best_estimator

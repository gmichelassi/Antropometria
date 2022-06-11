import os
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import time

from antropometria.preprocessing import run_preprocessing
from antropometria.utils.dataset.load import LoadData
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def run_random_forest_analysis():
    """
    Melhor RF
    Sem red. dim. | Filtro 0.98 | Sem min_max | Smote SVM
    Params: {'bootstrap': True, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 24,
    'min_samples_leaf': 1, 'n_estimators': 1500, 'n_jobs': -1, 'random_state': 707878}
    """
    x, y, classes = run_preprocessing(
        'dlibHOG', 'distances_all_px_eu', ['casos', 'controles'], False, 0.98, None, 'SVM', False
    )
    rf = RandomForestClassifier(
        n_estimators=1500,
        bootstrap=True,
        class_weight=None,
        criterion='gini',
        max_depth=None,
        max_features=24,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=707878
    )
    rf.fit(x, y)

    sel = SelectFromModel(rf, max_features=rf.n_features_)
    x_default, y_default = LoadData('dlibHOG', 'distances_all_px_eu', ['casos', 'controles']).load()
    sel.fit(x_default, y_default)
    columns = (x_default.loc[:, sel.get_support()]).columns.values

    feature_importances = pd.Series(rf.feature_importances_, index=columns)

    sorted_feature_importances = feature_importances.sort_values(ascending=False)
    best_features = sorted_feature_importances.head(15)

    best_features.to_csv('./antropometria/output/rf_best_features.csv')


if __name__ == '__main__':
    start_time = time.time()
    run_random_forest_analysis()
    print("--- Total execution time: %s minutes ---" % ((time.time() - start_time) / 60))

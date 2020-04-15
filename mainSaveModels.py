from classifiers import rf, svm, nb, nnn, knn
from sklearn.pipeline import make_pipeline
import pickle
import joblib

# Feature selection / Dimensionality reduction
from classifiers.custom_feature_selection import mRMRProxy, FCBFProxy, CFSProxy, RFSProxy
from sklearn.decomposition import PCA
from skrebate import ReliefF

import asd_data as asd

from classifiers.utils import apply_pearson_feature_selection
import pandas as pd
import numpy as np

from config import logger
import initContext as context
context.loadModules()
log = logger.getLogger(__file__)


def saveModel(filtro=0.0):
    all_samples = {}

    X, y = asd.load_data(d_type='euclidian', unit='px', m='', normalization='', dataset='all', labels=False)
    all_samples['euclidian_px_all'] = (X, y)

    for k in all_samples.keys():
        log.info("Generation final model for " + k + " dataset")
        log.info("Follow the steps to choose a classifier and save it")
        samples, labels = all_samples[k]
        if filtro != 0.0:
            samples = apply_pearson_feature_selection(samples, filtro)
        else:
            samples = samples.values

        instances, features = samples.shape
        n_features_to_keep = int(np.sqrt(features))

        while True:
            dimensionality_reduction = None
            estimator = None
            filename = ""

            print("1. Random Forest")
            print("2. SVM")
            print("3. GaussianNB")
            print("4. Neural Network")
            print("5. KNeighbors")
            print("0. Sair")
            classifier = int(input("Selecione um dos modelos para salvar: "))

            if 2 <= classifier <= 5:
                print("\n")
                print("1. PCA")
                print("2. mRMRProxy")
                print("3. FCBFProxy")
                print("4. CFSProxy")
                print("5. RFSProxy")
                print("6. ReliefF")
                print("0. Cancelar")
                dimensionality = int(input("Selecione um redutor de dimensionalidade: "))

                if dimensionality == 1:
                    dimensionality_reduction = PCA(n_components=n_features_to_keep, whiten=True)
                    filename = filename + "pca_"
                elif dimensionality == 2:
                    dimensionality_reduction = mRMRProxy(n_features_to_select=n_features_to_keep, verbose=False)
                    filename = filename + "mrmr_"
                elif dimensionality == 3:
                    dimensionality_reduction = FCBFProxy(n_features_to_select=n_features_to_keep, verbose=False)
                    filename = filename + "fcbf_"
                elif dimensionality == 4:
                    dimensionality_reduction = CFSProxy(n_features_to_select=n_features_to_keep, verbose=False)
                    filename = filename + "cfs_"
                elif dimensionality == 5:
                    dimensionality_reduction = RFSProxy(n_features_to_select=n_features_to_keep, verbose=False)
                    filename = filename + "rfs_"
                elif dimensionality == 6:
                    dimensionality_reduction = ReliefF(n_features_to_select=n_features_to_keep, n_neighbors=100, n_jobs=-1)
                    filename = filename + "reliff_"
                else:
                    dimensionality_reduction = None
                    pass

            if classifier == 1:
                estimator, name = rf.make_estimator()
                filename = filename + "rf"
            elif classifier == 2:
                estimator, name = svm.make_estimator()
                filename = filename + "svm"
            elif classifier == 3:
                filename = filename + "nb"
                estimator, name = nb.make_estimator()
            elif classifier == 4:
                estimator, name = nnn.make_estimator()
                filename = filename + "nnn"
            elif classifier == 5:
                estimator, name = knn.make_estimator()
                filename = filename + "knn"
            else:
                estimator = None
                pass

            if estimator is not None:
                pipe = make_pipeline(dimensionality_reduction, estimator)

                log.info("Training selected model")
                pipe.fit(samples, labels)  # training the model

                log.info("Saving selected model")
                # pickle.dump(pipe, open('{0}.sav'.format(filename), 'wb'))
                joblib.dump(pipe, '{0}.joblib'.format(filename))

                log.info("Done without errors!")
            else:
                log.info("It was not possible to save this model")
                break


if __name__ == '__main__':
    saveModel()

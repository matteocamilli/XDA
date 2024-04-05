from sklearn.decomposition import PCA
import numpy as np


def pcaClassifier(X_train, nComponents):
    pca = PCA(n_components=nComponents)
    pca.fit(X_train)

    explained_variance = pca.explained_variance_ratio_

    return explained_variance

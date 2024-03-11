from sklearn.decomposition import PCA
import numpy as np

def pcaClassifier(X_train, controllableFeatures):

    pca = PCA()
    pca.fit(X_train)

    explained_variance = pca.explained_variance_ratio_

    importance = np.abs(explained_variance)

    features_importance = list(zip(controllableFeatures, importance))

    features_importance.sort(key=lambda x: x[1], reverse=True)

    sorted_feature_indices = [feature[0] for feature in features_importance]

    print(sorted_feature_indices)

    return sorted_feature_indices
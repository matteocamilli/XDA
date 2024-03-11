import shap
import numpy as np


def shapClassifier(model, X_train, controllableFeatures):
    explainer = shap.Explainer(model)

    shap_values = explainer.shap_values(X_train)

    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    features_importance = list(zip(controllableFeatures, mean_shap_values))

    features_importance.sort(key=lambda x: x[1], reverse=True)

    sorted_feature_indices = [feature[0] for feature in features_importance]

    print(sorted_feature_indices)

    # shap.summary_plot(shap_values, plot_type='bar')

    return sorted_feature_indices

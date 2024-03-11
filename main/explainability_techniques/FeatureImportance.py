from sklearn.inspection import permutation_importance
import numpy as np


def permutation_importance_classifier(model, X_train, y_train, controllable_features):
    results = []
    for col_idx in range(y_train.shape[1]):
        y_train_col = y_train.iloc[:, col_idx].values
        result_col = permutation_importance(model, X_train, y_train_col, n_repeats=10, random_state=42)
        results.append(result_col)

    importances = np.mean([result.importances for result in results], axis=0)

    features_importance = list(zip(controllable_features, importances))

    features_importance.sort(key=lambda x: x[1][0], reverse=True)

    sorted_feature_indices = [feature[0] for feature in features_importance]

    print(sorted_feature_indices)

    return sorted_feature_indices

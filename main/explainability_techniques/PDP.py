from sklearn.inspection import PartialDependenceDisplay


def partial_dependence_plot(model, X_train, features):
    PartialDependenceDisplay.from_estimator(model, X_train, features)
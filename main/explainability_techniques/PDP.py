import matplotlib.pyplot as plt
import sklearn.inspection as ins
import seaborn as sns

def partial_dependence_plot(model, X_train, features, kind, pathName=None):
    pdp = ins.PartialDependenceDisplay.from_estimator(model, X_train, features, kind = kind)
    plt.tight_layout()
    if pathName is not None:
        plt.savefig(pathName)
    plt.show()
    return pdp

def partial_dependence_max_value(model, X_train, feature):
    pdp = partial_dependence_plot(model, X_train, [feature], "both")

    data = pdp.lines_[0, 0, 60].get_data(True)
    probs = data[1]
    var = data[0]
    max_point = 0
    for i in range(60):
        if probs[i] > probs[max_point]: max_point = i

    return var[max_point]
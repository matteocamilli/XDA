import matplotlib.pyplot as plt
import sklearn.inspection as ins


def partialDependencePlot(model, X_train, features, kind, path=None):
    pdp = ins.PartialDependenceDisplay.from_estimator(model, X_train, features, kind=kind)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    # plt.show()
    plt.clf()
    return pdp


def getMaxPoint(pdp):
    data = pdp.lines_[0, 0, 60].get_data(True)
    probs = data[1]
    var = data[0]
    max_point = 0
    for i in range(60):
        if probs[i] > probs[max_point]:
            max_point = i

    return var[max_point]


def getSlope(pdp, x):
    data = pdp.lines_[0, 0, 60].get_data(True)
    probs = data[1]
    var = data[0]
    index = len(var) - 1
    for i in range(len(var)):
        if var[i] > x:
            index = i
            break
    if index == 0:
        index = 1
    slope = (probs[i] - probs[index - 1]) / (var[index] - var[index - 1])
    return slope

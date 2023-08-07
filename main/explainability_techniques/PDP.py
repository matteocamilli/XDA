import matplotlib.pyplot as plt
import sklearn.inspection as ins
import seaborn as sns

def partial_dependence_plot(model, X_train, features, kind, pathName):
    ins.PartialDependenceDisplay.from_estimator(model, X_train, features, kind = kind)
    plt.tight_layout()
    plt.savefig(pathName)
    plt.show()

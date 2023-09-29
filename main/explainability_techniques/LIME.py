import matplotlib.pyplot as plt
import numpy as np
from lime import lime_tabular


def createLimeExplainer(X_train):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['bad', 'good'],
        mode='classification'
    )
    return explainer


def explain(explainer, model, data_row):
    exp = explainer.explain_instance(
        data_row,
        predict_fn=model.predict_proba
    )

    return exp


def saveExplanation(explanation, path=None):
    explanation.as_pyplot_figure()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    # plt.show()
    plt.clf()


def sortExplanation(explanation):
    local_exp = explanation.local_exp[1]
    local_exp.sort(key=lambda k: k[1], reverse=True)

    return local_exp

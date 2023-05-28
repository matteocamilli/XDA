import matplotlib.pyplot as plt
import numpy as np
from lime import lime_tabular

def createLIMEExplainer(X_train):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data = np.array(X_train),
        feature_names = X_train.columns,
        class_names = ['bad', 'good'],
        mode = 'classification'
    )
    return explainer

def explain(explainer, model, data_row):
    exp = explainer.explain_instance(
        data_row,
        predict_fn = model.predict_proba
    )
    #exp.show_in_notebook(show_table = True)

    fig = exp.as_pyplot_figure()

    plt.tight_layout()
    plt.show()

import shap
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt



def shapClassifier(model, X_train, controllableFeatures):
    if isinstance(model, LogisticRegression):
        explainer = shap.KernelExplainer(model.predict, X_train)

    elif isinstance(model, RandomForestClassifier):

        explainer = shap.TreeExplainer(model)

    elif isinstance(model, DecisionTreeClassifier):

        explainer = shap.TreeExplainer(model)

    elif isinstance(model, MLPClassifier):

        explainer = shap.KernelExplainer(model.predict, X_train)

    elif isinstance(model, GradientBoostingClassifier):

        explainer = shap.Explainer(model)

    elif isinstance(model, XGBClassifier):
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_train)

    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    features_importance = list(zip(controllableFeatures, mean_shap_values))

    scores = np.array([np.sum(tup[1]) for tup in features_importance])
    
    return scores
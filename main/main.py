import pandas as pd
from numpy import ravel
from sklearn.model_selection import train_test_split

from explainability_techniques.LIME import createLIMEExplainer, explain
from explainability_techniques.PDP import partial_dependence_plot
from model.ModelConstructor import constructModel

if __name__ == '__main__':

    ds = pd.read_csv('datasets/data.csv')
    features = ["power",
                "cruise speed",
                "illuminance",
                "smoke intensity",
                "obstacle size",
                "obstacle distance",
                "firm obstacle"]
    outcomes = ["req_0"]
    # , "req_1", "req_2", "req_3", "req_4", "req_5"
    X = ds.loc[:, features]
    y = ds.loc[:, outcomes]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    y_test = ravel(y_test)
    y_train = ravel(y_train)

    models = constructModel(X_train, X_test, y_train, y_test)

    explainer = createLIMEExplainer(X_train)

    data_row = X_test.iloc[50]
    for m in models:
        explain(explainer, m, data_row)
        partial_dependence_plot(m, X_train, ["power", "cruise speed", ("power", "cruise speed")])

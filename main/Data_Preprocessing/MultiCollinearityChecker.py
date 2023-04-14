import pandas as pd
import random
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def multiCheck():
    ds = pd.read_csv('Datasets/data1.csv')
    features = ["power",
                "cruise speed",
                "bandwidth",
                "quality",
                "illuminance",
                "smoke intensity",
                "obstacle size",
                "obstacle distance",
                "firm obstacle"]
    X = ds.loc[:, features]

    # Prepare a dataframe for VIF
    X_VIF = add_constant(X)

    # Calculate VIF scores
    vif_scores = pd.DataFrame([variance_inflation_factor(X_VIF.values, i)
                   for i in range(X_VIF.shape[1])],
                  index=X_VIF.columns)
    # Prepare a final dataframe of VIF scores
    vif_scores.reset_index(inplace = True)
    vif_scores.columns = ['Feature', 'VIFscore']
    vif_scores = vif_scores.loc[vif_scores['Feature'] != 'const', :]
    vif_scores = vif_scores.sort_values(by = ['VIFscore'], ascending = False)
    print(vif_scores)
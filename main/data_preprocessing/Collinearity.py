from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def collCheck(X):

    # There are 3 options for the parameter setting of method as follows:
    # pearson : standard correlation coefficient
    # kendall : Kendall Tau correlation coefficient
    # spearman : Spearman rank correlation
    corrmat = X.corr(method='spearman')
    top_corr_features = corrmat.index

    # Visualise a lower-triangle correlation heatmap
    mask_df = np.triu(np.ones(corrmat.shape)).astype(bool)
    plt.figure(figsize=(10, 8))
    # plot heat map
    g = sns.heatmap(X[top_corr_features].corr(),
                    mask=mask_df,
                    vmin=-1,
                    vmax=1,
                    annot=True,
                    cmap="RdBu")

    plt.show()

def multiCheck(X):

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
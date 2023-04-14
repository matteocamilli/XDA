import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def collCheck():
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
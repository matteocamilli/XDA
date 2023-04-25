# Import for Construct Defect Models (Classification)
import numpy as np
import pandas as pd
from numpy import ravel
from IPython.display import display
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.ensemble import RandomForestClassifier # Random Forests
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # C5.0 (Decision Tree)
from sklearn.neural_network import MLPClassifier # Neural Network
from sklearn.ensemble import GradientBoostingClassifier # Gradient Boosting Machine (GBM)
import xgboost as xgb # eXtreme Gradient Boosting Tree (xGBTree)

# Import for AUC calculation
from sklearn.metrics import roc_auc_score

def constructModel():

    ds = pd.read_csv('datasets/data1.csv')
    features = ["power",
                "cruise speed",
                "bandwidth",
                "quality",
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

    ## Construct defect models

    # Logistic Regression
    lr_model = LogisticRegression(random_state=1234)
    lr_model.fit(X_train, y_train)
    lr_model_AUC = round(roc_auc_score(y_test, lr_model.predict_proba(X_test)[:,1]), 3)

    # Random Forests
    rf_model = RandomForestClassifier(random_state=1234, n_jobs = 10)
    rf_model.fit(X_train, y_train)
    rf_model_AUC = round(roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]), 3)

    # C5.0 (Decision Tree)
    dt_model = DecisionTreeClassifier(random_state=1234)
    dt_model.fit(X_train, y_train)
    dt_model_AUC = round(roc_auc_score(y_test, dt_model.predict_proba(X_test)[:,1]), 3)

    # Neural Network
    nn_model = MLPClassifier(random_state=1234)
    nn_model.fit(X_train, y_train)
    nn_model_AUC = round(roc_auc_score(y_test, nn_model.predict_proba(X_test)[:,1]), 3)

    # Gradient Boosting Machine (GBM)
    gbm_model = GradientBoostingClassifier(random_state=1234)
    gbm_model.fit(X_train, y_train)
    gbm_model_AUC = round(roc_auc_score(y_test, gbm_model.predict_proba(X_test)[:,1]), 3)

    # eXtreme Gradient Boosting Tree (xGBTree)
    xgb_model = xgb.XGBClassifier(random_state=1234)
    xgb_model.fit(X_train, y_train)
    xgb_model_AUC = round(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]), 3)

    # Summarise into a DataFrame
    model_performance_df = pd.DataFrame(data=np.array([['Logistic Regression', 'Random Forests', 'C5.0 (Decision Tree)', 'Neural Network', 'Gradient Boosting Machine (GBM)', 'eXtreme Gradient Boosting Tree (xGBTree)'],
                [lr_model_AUC, rf_model_AUC, dt_model_AUC, nn_model_AUC, gbm_model_AUC, xgb_model_AUC]]).transpose(),
                index = range(6),
                columns = ['Model', 'AUC'])
    model_performance_df['AUC'] = model_performance_df.AUC.astype(float)
    model_performance_df = model_performance_df.sort_values(by = ['AUC'], ascending = False)

    # Visualise the performance of defect models
    display(model_performance_df)
    model_performance_df.plot(kind='barh', y='AUC', x='Model')
import os
import pandas as pd
import numpy as np
from numpy import ravel
from sklearn.model_selection import train_test_split

from explainability_techniques.LIME import createLIMEExplainer, explain
from explainability_techniques.PDP import partial_dependence_plot, partial_dependence_plot
from model.ModelConstructor import constructModel

if __name__ == '__main__':

    ds = pd.read_csv('../datasets/data.csv')
    features = ["cruise speed",
                #"image resolution",
                #"flashlight intensity",
                #"suspension responsiveness",
                "power",
                "illuminance",
                "smoke intensity",
                "obstacle size",
                "obstacle distance",
                "firm obstacle"]
    outcomes = ["req_4"]            # , "req_1", "req_2", "req_3", "req_4", "req_5"
    X = ds.loc[:, features]
    y = ds.loc[:, outcomes]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    y_test = ravel(y_test)
    y_train = ravel(y_train)

    models = constructModel(X_train, X_test, y_train, y_test)

    """
    couplesOfFeatures = []
    featureToCycles = features.copy()
    for f1 in features:
        featureToCycles.remove(f1)
        for f2 in featureToCycles:
            couplesOfFeatures.append((f1, f2))
            
    for m in models:
        path = '../plots/' + m.__class__.__name__
        if not os.path.exists(path): os.makedirs(path)
        
        for f in features:
            path = 'plots/' + m.__class__.__name__ + '/individuals'
            if not os.path.exists(path): os.makedirs(path)
            partial_dependence_plot(m, X_train, [f], "both", path + '/' + f + '.png')
            
        for c in couplesOfFeatures:
            path = 'plots/' + m.__class__.__name__ + '/couples'
            if not os.path.exists(path): os.makedirs(path)
            partial_dependence_plot(m, X_train, [c], "average", path + '/' + c[0] + ' % ' + c[1] + '.png')
    """


    explainer = createLIMEExplainer(X_train)
    data_row = X_test.iloc[50]

    for m in models:
        explaination = explain(explainer, m, data_row)
        local_exp = explaination.local_exp[1]
        local_exp.sort(key=lambda k: k[0])
        for i in range(len(features)):
            local_exp[i] = (features[i], local_exp[i][1])
        local_exp.sort(key=lambda k: k[1])
        #print(local_exp)



    os.chdir("../MDP_Dataset_Builder")
    mod_dataset = X.to_numpy(copy=True)                         #.loc[0:10, :]. to test only part of the dataset
    np.save("./starting_combinations.npy", mod_dataset)
    os.system("execute.bat ./starting_combinations.npy")
    os.system("merge_csvs.py")

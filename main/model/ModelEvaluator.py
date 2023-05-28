from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def evaluatePrecision(rf_model, X_test, y_test):
    rf_precision_function = precision_score(y_test, rf_model.predict(X_test))
    print('Precision (precision_score function)\t:', rf_precision_function)
    return rf_precision_function

def evaluateRecall(rf_model, X_test, y_test):
    rf_recall_function = recall_score(y_test, rf_model.predict(X_test))
    print('Recall (recall_score function)\t:', rf_recall_function)
    return rf_recall_function

def evaluateFAR(rf_model, X_test, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, rf_model.predict(X_test)).ravel()
    rf_FAR_manual = fp / (fp + tn)
    print('FAR (manual calculation)\t\t:', rf_FAR_manual)
    return rf_FAR_manual

def evaluateAUC(rf_model, X_test, y_test):
    rf_AUC_function = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    print('AUC (roc_auc_score function)\t\t:', rf_AUC_function)


def evaluateMCC(rf_model, X_test, y_test):
    rf_MCC_function = matthews_corrcoef(y_test, rf_model.predict(X_test))
    print('MCC (matthews_corrcoef function)\t:', rf_MCC_function)

def evaluateMCC(rf_model, X_test, y_test):
    # Generate a defect-proneness ranking of testing instances
    X_test_df = X_test.copy()
    X_test_df['predicted_prob'] = rf_model.predict_proba(X_test)[:, 1]
    X_test_df = X_test_df.sort_values(by=['predicted_prob'], ascending=False)

    # Determine the Initial False Alarm (IFA)
    IFA = 0
    for test_index in X_test_df.index:
        IFA += 1
        if y_test.loc[test_index] == 1:
            break

    print('Initial False Alarm (IFA)\t\t:', IFA)


def evaluateDTH(rf_model, X_test, y_test):
    # calculate recall with a function
    rf_recall_function = recall_score(y_test, rf_model.predict(X_test))

    # calculate FAR manually
    tn, fp, fn, tp = confusion_matrix(y_test, rf_model.predict(X_test)).ravel()
    rf_FAR_manual = fp / (fp + tn)

    # calculate D2H manually
    rf_D2H_numerator = ((1 - rf_recall_function) ** 2) + ((0 - rf_FAR_manual) ** 2)
    rf_D2H_denominator = 2
    rf_D2H_manual = (rf_D2H_numerator / rf_D2H_denominator) ** (1 / 2)

    print('D2H (manual calculation)\t\t:', rf_D2H_manual)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def evaluatePrecision(rf_model, X_test, y_test):

    # construct a confusion matrix
    print('Construct a confusion matrix:')
    tn, fp, fn, tp = confusion_matrix(y_test, rf_model.predict(X_test)).ravel()
    print('(True Positive, False Positive) = (' + str(tp) +','+str(fp)+')')
    print('(False Negative, True Negative) = (' + str(fn) +','+str(tn)+')\n')

    # calculate precision manually
    rf_precision_manual = tp/(tp + fp)

    # calculate precision with a function
    rf_precision_function = precision_score(y_test, rf_model.predict(X_test))

    print('Precision (manual calculation)\t\t:', rf_precision_manual)
    print('Precision (precision_score function)\t:', rf_precision_function)


def evaluateRecall(rf_model, X_test, y_test):
    # calculate recall manually
    #rf_recall_manual = tp / (tp + fn)

    # calculate recall with a function
    rf_recall_function = recall_score(y_test, rf_model.predict(X_test))

    #print('Recall (manual calculation)\t\t:', rf_recall_manual)
    print('Recall (recall_score function)\t:', rf_recall_function)
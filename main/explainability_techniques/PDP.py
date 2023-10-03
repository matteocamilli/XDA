import matplotlib.pyplot as plt
import sklearn.inspection as ins
import copy

def partialDependencePlot(model, X_train, features, kind, path=None):
    pdp = ins.PartialDependenceDisplay.from_estimator(model, X_train, features, kind=kind)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    # plt.show()
    plt.clf()
    return pdp

def multiplyPdps(pdps, path=None):
    res = copy.deepcopy(pdps[0])
    for pdp in pdps[1:]:
        for i in range(len(res.pd_results[0]["individual"][0])):
            res.pd_results[0]["individual"][0][i, :] *= pdp.pd_results[0]["individual"][0][i, :]
        res.pd_results[0]["average"][0, :] *= pdp.pd_results[0]["average"][0, :]
    res.plot()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    # plt.show()
    plt.clf()
    return res

def getMaxPointOfLine(pdp, lineIndex):
    if lineIndex >= 0:
        yVals = pdp.pd_results[0]["individual"][0][lineIndex, :]
    else:
        yVals = pdp.pd_results[0]["average"][0, :]
    xVals = pdp.pd_results[0]["values"][0]
    max_point = 0
    for i in range(len(yVals)):
        if yVals[i] > yVals[max_point]:
            max_point = i

    return xVals[max_point]

def getMaxPointOfMeanLine(pdp):
    return getMaxPointOfLine(pdp, -1)

def getMaxPointOfClosestLine(pdp, x, y):
    return getMaxPointOfLine(pdp, getClosestLine(pdp, x, y))

def getClosestLine(pdp, x, y):
    xVals = pdp.pd_results[0]["values"][0]
    closestXIndex = binarySearchClosestIndex(x, xVals)  # same for each line

    linesY = pdp.pd_results[0]["individual"][0][:, closestXIndex]

    distance = -1
    lineIndex = -1
    for i in range(len(linesY)):
        newDist = abs(linesY[i] - y)
        if newDist < distance or distance == -1:
            distance = newDist
            lineIndex = i
    return lineIndex

def binarySearchClosest(elem, list):
    if len(list) == 1:
        return list[0]
    else:
        sublist1 = list[:len(list)//2]
        sublist2 = list[len(list)//2:]
        el1 = sublist1[len(sublist1) - 1]
        el2 = sublist2[0]
        if elem == el1:
            return el1
        elif abs(elem - el1) < abs(elem - el2):
            return binarySearchClosest(elem, sublist1)
        else:
            return binarySearchClosest(elem, sublist2)

def binarySearchClosestIndex(elem, list):
    sublist1 = list[:len(list)//2]
    sublist2 = list[len(list)//2:]
    index = len(sublist1) - 1
    el1 = sublist1[index]
    el2 = sublist2[0]
    if elem == el1:
        return index
    elif abs(elem - el1) < abs(elem - el2):
        return binarySearchClosestIndexHelper(elem, sublist1, index)
    else:
        sublist2 = list[len(list)//2:]
        return binarySearchClosestIndexHelper(elem, sublist2, len(list) - 1)

def binarySearchClosestIndexHelper(elem, list, index):
    if len(list) == 1:
        return index
    else:
        sublist1 = list[:len(list)//2]
        sublist2 = list[len(list)//2:]
        i = index - (len(list) - len(sublist1))
        el1 = sublist1[len(sublist1) - 1]
        el2 = sublist2[0]
        if elem == el1:
            return i
        elif abs(elem - el1) < abs(elem - el2):
            return binarySearchClosestIndexHelper(elem, sublist1, i)
        else:
            return binarySearchClosestIndexHelper(elem, list[len(list)//2:], index)

def getSlope(pdp, x, lineIndex):
    xVals = pdp.pd_results[0]["values"][0]
    linesY = pdp.pd_results[0]["individual"][0][lineIndex, :]
    index = len(xVals) - 1
    for i in range(len(xVals)):
        if xVals[i] > x:
            index = i
            break
    if index == 0:
        index = 1
    slope = (linesY[i] - linesY[index - 1]) / (xVals[index] - xVals[index - 1])
    return slope

def getSlopeOfClosestLine(pdp, x, y):
    return getSlope(pdp, x, getClosestLine(pdp, x, y))

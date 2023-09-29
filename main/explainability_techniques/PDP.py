import matplotlib.pyplot as plt
import sklearn.inspection as ins


def partialDependencePlot(model, X_train, features, kind, path=None):
    pdp = ins.PartialDependenceDisplay.from_estimator(model, X_train, features, kind=kind)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    # plt.show()
    plt.clf()
    return pdp

def getMaxPointOfLine(pdp, lineIndex):
    data = pdp.lines_[0, 0, lineIndex].get_data(True)
    xVals = data[0]
    yVals = data[1]
    max_point = 0
    for i in range(60):
        if yVals[i] > yVals[max_point]:
            max_point = i

    return xVals[max_point]

def getMaxPointOfMeanLine(pdp):
    return getMaxPointOfLine(pdp, 60)

def getMaxPointOfClosestLine(pdp, x, y):
    line = pdp.lines_[0, 0, 0].get_data(True)
    xVals = line[0]
    closestXIndex = binarySearchClosestIndex(x, xVals)  # same for each line

    linesY = []
    for i in range(60):
        line = pdp.lines_[0, 0, i].get_data(True)
        yVals = line[1]
        linesY.append(yVals[closestXIndex])

    distance = -1
    lineIndex = -1
    for i in range(len(linesY)):
        newDist = abs(linesY[i] - y)
        if newDist < distance or distance == -1:
            distance = newDist
            lineIndex = i
    return getMaxPointOfLine(pdp, lineIndex)

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

def getSlope(pdp, x):
    data = pdp.lines_[0, 0, 60].get_data(True)
    probs = data[1]
    var = data[0]
    index = len(var) - 1
    for i in range(len(var)):
        if var[i] > x:
            index = i
            break
    if index == 0:
        index = 1
    slope = (probs[i] - probs[index - 1]) / (var[index] - var[index - 1])
    return slope

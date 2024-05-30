import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import PercentFormatter, FuncFormatter, FormatStrFormatter

from util import readFromCsv, evaluateAdaptations

font = {'family': 'sans',
        'weight': 'normal',
        'size': 12}

matplotlib.rc('font', **font)


def personalizedBoxPlot(data, name, columnNames=None, percentage=False, path=None, show=False, seconds=False,
                        legendInside=False):
    fig = plt.figure(figsize=(15, 8))

    ax1 = fig.add_subplot(111)
    bp = ax1.boxplot(data, patch_artist=True, notch=True, vert=True)

    # Definizione dei colori per gli algoritmi
    algorithm_colors = [
        '#FF5733',
        '#6B8E23',
        '#4169E1',
    ]

    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(algorithm_colors[i % 3])

    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B', linewidth=1.5, linestyle=":")

    for cap in bp['caps']:
        cap.set(color='#8B008B', linewidth=2)

    for median in bp['medians']:
        median.set(color='red', linewidth=3)

    for flier in bp['fliers']:
        flier.set(marker='D', color='#e7298a', alpha=0.5)

    if columnNames is not None and len(columnNames) > 1:
        ax1.xaxis.set_ticks(np.arange(2, len(columnNames) * 6, step=6), columnNames)
    else:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    if percentage:
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

        if (data.max().max() - data.min().min()) / 8 < 0.01:
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

    if seconds:
        def y_fmt(x, y):
            return str(int(x)) + ' s'

        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

    ax1.set_yscale('log')

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    if legendInside:
        ax1.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2]],
                   ["XDA", "XDA SHAP", "XDA FI"])
    else:
        ax1.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2]],
                   ["XDA", "XDA SHAP", "XDA FI"],
                   ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))

    plt.title(name)

    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    if path is not None:
        plt.savefig(path + name)

    if show:
        fig.show()
    else:
        plt.clf()


def personalizedBoxPlotCustom(data, name, columnNames=None, percentage=False, path=None, show=False, seconds=False,
                              legendInside=False):
    fig = plt.figure(figsize=(15, 8))  # 1500x800

    ax1 = fig.add_subplot(111)
    bp = ax1.boxplot(data, patch_artist=True, notch=True, vert=True)

    algorithm_colors = [
        '#FF5733',
        '#6B8E23',
        '#4169E1',
    ]

    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(algorithm_colors[i % 3])

    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B', linewidth=1.5, linestyle=":")

    for cap in bp['caps']:
        cap.set(color='#8B008B', linewidth=2)

    for median in bp['medians']:
        median.set(color='red', linewidth=3)

    for flier in bp['fliers']:
        flier.set(marker='D', color='#e7298a', alpha=0.5)

    if percentage:
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

        if (data.max().max() - data.min().min()) / 8 < 0.01:
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

    if seconds:
        def y_fmt(x, y):
            return str(int(x)) + ' s'

        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

    ax1.set_yscale('log')

    num_boxes = len(data.T)
    num_groups = (num_boxes + 2) // 3
    group_labels = ['R' + str(i + 1) for i in range(num_groups)]
    positions = [3 * i + 2 for i in range(num_groups)]

    ax1.set_xticks(positions)
    ax1.set_xticklabels(group_labels, rotation=0, fontsize=12)

    # Add vertical lines to separate groups
    for i in range(1, num_groups):
        plt.axvline(x=3 * i + 0.5, color='gray', linestyle='--', linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    if legendInside:
        ax1.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2]],
                   ["XDA", "XDA SHAP", "XDA FI"])
    else:
        ax1.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2]],
                   ["XDA", "XDA SHAP", "XDA FI"],
                   ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))

    plt.title(name)

    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    if path is not None:
        plt.savefig(path + name)

    if show:
        plt.show()
    else:
        plt.clf()


def personalizedBarChart(data, name, path=None, show=False, percentage=False):
    colors = plt.cm.Spectral(np.linspace(0, 1, 5))

    ax = data.plot.bar(title=name, color=colors, figsize=(15, 8))

    plt.legend()

    if len(data.index) > 1:
        plt.xticks(rotation=0)
    else:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax.set_ylim(0, 1)
    if percentage:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    for container in ax.containers:
        if percentage:
            values = ['{:.1%}'.format(v) for v in container.datavalues]
        else:
            values = ['{:.2}'.format(v) for v in container.datavalues]
        ax.bar_label(container, values, fontsize=10)

    ax.legend(loc='lower right')

    if path is not None:
        plt.savefig(path + name)

    if show:
        plt.show()
    else:
        plt.clf()


os.chdir(sys.path[0])
evaluate = False

pathToResults = "../results/autonomousDrivingv3/"

featureNames = ["car_speed",
                    "p_x",
                    "p_y",
                    "orientation",
                    "weather",
                    "road_shape"]

reqs = ["req_0", "req_1", "req_2"]
reqsNamesInGraphs = ["R1", "R2", "R3"]

# read dataframe from csv
results = readFromCsv(pathToResults + 'results.csv')
resultsSHAP = readFromCsv(pathToResults + 'resultsSHAP.csv')
resultsFI = readFromCsv(pathToResults + 'resultsFI.csv')
nReqs = len(results["custom_confidence"][0])
reqs = reqs[:nReqs]
reqsNamesInGraphs = reqsNamesInGraphs[:nReqs]
targetConfidence = np.full((1, nReqs), 0.8)[0]

if evaluate:
    evaluateAdaptations(results, resultsSHAP, resultsFI, featureNames)

# read outcomes from csv
customOutcomes = pd.read_csv(pathToResults + 'customDataset.csv')
SHAPOutcomes = pd.read_csv(pathToResults + 'SHAPDataset.csv')
FIOutcomes = pd.read_csv(pathToResults + 'FIDataset.csv')
#nsga3Outcomes = pd.read_csv(pathToResults + 'nsga3Dataset.csv')

# build indices arrays
#nsga3ConfidenceNames = ['nsga3_confidence_' + req for req in reqs]
#nsga3OutcomeNames = ['nsga3_outcome_' + req for req in reqs]
customConfidenceNames = ['custom_confidence_' + req for req in reqs]
customOutcomeNames = ['custom_outcome_' + req for req in reqs]
SHAPcustomConfidenceNames = ['custom_confidence_' + req for req in reqs]
SHAPcustomOutcomeNames = ['custom_outcome_' + req for req in reqs]
FIcustomConfidenceNames = ['custom_confidence_' + req for req in reqs]
FIcustomOutcomeNames = ['custom_outcome_' + req for req in reqs]

#outcomes dataframe
outcomes = pd.concat([customOutcomes[reqs]], axis=1)
outcomes.columns = customOutcomeNames
outcomes = outcomes[list(sum(zip(customOutcomeNames), ()))]
outcomesSHAP = SHAPOutcomes[reqs]
outcomesSHAP.columns = np.array(SHAPcustomOutcomeNames)
outcomesSHAP = outcomesSHAP[list(sum(zip(SHAPcustomOutcomeNames), ()))]
outcomesFI = FIOutcomes[reqs]
outcomesFI.columns = np.array(FIcustomOutcomeNames)
outcomesFI = outcomesFI[list(sum(zip(FIcustomOutcomeNames), ()))]

# decompose arrays columns into single values columns
#nsga3Confidences = pd.DataFrame(results['nsga3_confidence'].to_list(),
#                               columns=nsga3ConfidenceNames)
customConfidences = pd.DataFrame(results['custom_confidence'].to_list(),
                                 columns=customConfidenceNames)
customConfidencesSHAP = pd.DataFrame(resultsSHAP['custom_confidence'].to_list(),
                                     columns=SHAPcustomConfidenceNames)
customConfidencesFI = pd.DataFrame(resultsFI['custom_confidence'].to_list(),
                                   columns=FIcustomConfidenceNames)

# select sub-dataframes to plot
confidences = customConfidences
confidences = confidences[list(sum(zip(customConfidences.columns), ()))]
confidences_concat = pd.concat(
    [customConfidences, customConfidencesSHAP, customConfidencesFI], axis=1)
#confidences_concat = confidences_concat[list(sum(zip(nsga3Confidences.columns, customConfidences.columns, customConfidencesSHAP.columns, customConfidencesPCA.columns, customConfidencesFI.columns), ()))]
scores = pd.concat([results["custom_score"], resultsSHAP["custom_score"],
                    resultsFI["custom_score"]], axis=1)
times = pd.concat([results["custom_time"],
                   resultsSHAP["custom_time"],
                   resultsFI["custom_time"]], axis=1)
confidencesSHAP = pd.concat([customConfidencesSHAP], axis=1)
confidencesSHAP = confidencesSHAP[list(sum(zip(customConfidencesSHAP.columns), ()))]
scoresSHAP = resultsSHAP[["custom_score"]]
timesSHAP = resultsSHAP[["custom_time"]]
confidencesFI = pd.concat([customConfidencesFI], axis=1)
confidencesFI = confidencesFI[list(sum(zip(customConfidencesFI.columns), ()))]
scoresFI = resultsSHAP[["custom_score"]]
timesFI = resultsSHAP[["custom_time"]]

# plots
plotPath = pathToResults + 'plots/'
if not os.path.exists(plotPath):
    os.makedirs(plotPath)

personalizedBoxPlotCustom(confidences_concat, "Confidences comparison", reqsNamesInGraphs, path=plotPath,
                          percentage=False)
personalizedBoxPlot(scores, "Score comparison", path=plotPath)
personalizedBoxPlot(times, "Execution time comparison", path=plotPath, seconds=True, legendInside=True)

# predicted successful adaptations
#nsga3PredictedSuccessful = (confidences[nsga3ConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessful = (confidences[customConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulSHAP = (confidencesSHAP[SHAPcustomConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulFI = (confidencesFI[FIcustomConfidenceNames] > targetConfidence).all(axis=1)
predicted_successful_combined = pd.DataFrame({
    #'nsga3PredictedSuccessful': nsga3PredictedSuccessful,
    'customPredictedSuccessful': customPredictedSuccessful,
    'customPredictedSuccessfulSHAP': customPredictedSuccessfulSHAP,
    'customPredictedSuccessfulFI': customPredictedSuccessfulFI
})

#personalizedBoxPlot(confidences_concat[nsga3PredictedSuccessful], "Confidences comparison on NSGA-III predicted success", reqsNamesInGraphs, path=plotPath, percentage=False)
#personalizedBoxPlot(scores[nsga3PredictedSuccessful], "Score comparison on NSGA-III predicted success", path=plotPath)
#personalizedBoxPlot(times[nsga3PredictedSuccessful], "Execution time comparison on NSGA-III predicted success",
#                    path=plotPath, seconds=True, legendInside=True)

#print("NSGA-III predicted success rate: " + "{:.2%}".format(
#    nsga3PredictedSuccessful.sum() / nsga3PredictedSuccessful.shape[0]))
#print(str(nsga3Confidences.mean()) + "\n")
print("XDA predicted success rate:  " + "{:.2%}".format(
    customPredictedSuccessful.sum() / customPredictedSuccessful.shape[0]))
print(str(customConfidences.mean()) + "\n")
print("XDA SHAP predicted success rate:  " + "{:.2%}".format(
    customPredictedSuccessfulSHAP.sum() / customPredictedSuccessfulSHAP.shape[0]))
print(str(customConfidencesSHAP.mean()) + "\n")
print("XDA FI predicted success rate:  " + "{:.2%}".format(
    customPredictedSuccessfulFI.sum() / customPredictedSuccessfulFI.shape[0]))
print(str(customConfidencesFI.mean()) + "\n")

#print("NSGA-III mean probas of predicted success: \n" + str(nsga3Confidences[nsga3PredictedSuccessful].mean()) + '\n')
print("XDA mean probas of predicted success: \n" + str(customConfidences[customPredictedSuccessful].mean()) + '\n')
print("XDA SHAP mean probas of predicted success: \n" + str(
    customConfidencesSHAP[customPredictedSuccessfulSHAP].mean()) + '\n')
print(
    "XDA FI mean probas of predicted success: \n" + str(customConfidencesFI[customPredictedSuccessfulFI].mean()) + '\n')

# predicted successful adaptations
#nsga3Successful = outcomes[nsga3OutcomeNames].all(axis=1)
customSuccessful = outcomes[customOutcomeNames].all(axis=1)
customSuccessfulSHAP = outcomesSHAP[customOutcomeNames].all(axis=1)
customSuccessfulFI = outcomesFI[customOutcomeNames].all(axis=1)

#nsga3SuccessRate = nsga3Successful.mean()
customSuccessRate = customSuccessful.mean()
customSuccessRateSHAP = customSuccessfulSHAP.mean()
customSuccessRateFI = customSuccessfulFI.mean()

# outcomes analysis
#print("NSGA-III success rate: " + "{:.2%}".format(nsga3SuccessRate))
#print(str(outcomes[nsga3OutcomeNames].mean()) + "\n")
print("XDA success rate:  " + "{:.2%}".format(customSuccessRate))
print(str(outcomes[customOutcomeNames].mean()) + "\n")
print("XDA SHAP success rate:  " + "{:.2%}".format(customSuccessRateSHAP))
print(str(outcomesSHAP[SHAPcustomOutcomeNames].mean()) + "\n")
print("XDA FI success rate:  " + "{:.2%}".format(customSuccessRateFI))
print(str(outcomesFI[FIcustomOutcomeNames].mean()) + "\n")

successRateIndividual = pd.concat(
    [outcomes[customOutcomeNames].rename(columns=dict(zip(customOutcomeNames, reqsNamesInGraphs))).mean(),
     outcomesSHAP[SHAPcustomOutcomeNames].rename(columns=dict(zip(SHAPcustomOutcomeNames, reqsNamesInGraphs))).mean(),
     outcomesFI[customOutcomeNames].rename(columns=dict(zip(FIcustomOutcomeNames, reqsNamesInGraphs))).mean()],
    axis=1)
successRateIndividual.columns = ['XDA', 'XDA SHAP', 'XDA FI']
personalizedBarChart(successRateIndividual, "Success Rate Individual Reqs", plotPath)

successRate = pd.DataFrame(
    [[customSuccessRate, customSuccessRateSHAP, customSuccessRateFI]],
    columns=["XDA", "XDA SHAP", "XDA FI"])
personalizedBarChart(successRate, "Success Rate", plotPath)

successRateOfPredictedSuccess = pd.DataFrame([[outcomes[customOutcomeNames][customPredictedSuccessful].all(
    axis=1).mean(),
                                               outcomesSHAP[SHAPcustomOutcomeNames][customPredictedSuccessfulSHAP].all(
                                                   axis=1).mean(),
                                               outcomesFI[FIcustomOutcomeNames][customPredictedSuccessfulFI].all(
                                                   axis=1).mean()]],
                                             columns=["XDA", "XDA SHAP", "XDA FI"])
personalizedBarChart(successRateOfPredictedSuccess, "Success Rate of Predicted Success", plotPath)

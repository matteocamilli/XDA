import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from util import readFromCsv, evaluateAdaptations


font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)


def personalizedBoxPlot(data, name, columnNames=None, percentage=False, path=None, show=False, seconds=False, legendInside=False):
    columns = data.columns
    nColumns = len(columns)
    fig = plt.figure()  # plt.figure(figsize=(10, 10 * nColumns/2))
    ax1 = fig.add_subplot(111)  # (nColumns, 1, 1)

    # Creating axes instance
    bp = ax1.boxplot(data, patch_artist=True,
                     notch='True', vert=True)

    colors = plt.cm.Spectral(np.linspace(.1, .9, 2))
    # colors = np.append(colors[0::2], colors[1::2], axis=0)
    c = np.copy(colors)
    for i in range(nColumns//2):
        c = np.append(c, colors, axis=0)

    colors = c

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B',
                    linewidth=1.5,
                    linestyle=":")

    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color='#8B008B',
                linewidth=2)

    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color='red',
                   linewidth=3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker='D',
                  color='#e7298a',
                  alpha=0.5)

    # x-axis labels
    if columnNames is not None and len(columnNames) > 1:
        ax1.xaxis.set_ticks(np.arange(1.5, len(columnNames) * 2, step=2), columnNames)
    else:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # y-axis
    if percentage:
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

        if (data.max().max() - data.min().min())/8 < 0.01:
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    if seconds:
        def y_fmt(x, y):
            return str(int(x)) + ' s'
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

    # legend
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    if legendInside:
        ax1.legend([bp["boxes"][0], bp["boxes"][1]], ["NSGA-III", "XDA"],)
    else:
        ax1.legend([bp["boxes"][0], bp["boxes"][1]], ["NSGA-III", "XDA"],
                   ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))

    # Adding title
    plt.title(name)

    # Removing top axes and right axes
    # ticks
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    """
    for i in range(int(nColumns/2)):
        i2 = i + int(nColumns/2)
        axn = fig.add_subplot(nColumns, 1, i + 2)
        subset = data[[columns[i], columns[i2]]]
        subset = subset.sort_values(columns[i2])
        subset = subset.reset_index(drop=True)
        # axn.title.set_text(columns[i] + ' | ' + columns[i + int(nColumns/2)])
        subset.plot(ax=axn, color=colors[[i, i2]])
    """

    if path is not None:
        plt.savefig(path + name)

    if show:
        fig.show()
    else:
        plt.clf()


def personalizedBarChart(data, name, path=None, show=False, percentage=False):
    colors = plt.cm.Spectral(np.linspace(.1, .9, 2))
    # colors = np.append(colors[0::2], colors[1::2], axis=0)
    c = np.copy(colors)
    for i in range(len(data.values) // 2):
        c = np.append(c, colors, axis=0)

    colors = c

    ax = data.plot.bar(title=name, color=colors)

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

    """
    for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 5,
            f"{rect.get_height() * 100:.1f}%", fontsize=7,
            ha='center', va='bottom')
    """

    if path is not None:
        plt.savefig(path + name)

    if show:
        plt.show()
    else:
        plt.clf()

os.chdir(sys.path[0])
evaluate = False

pathToResults = "../results/"

featureNames = ["formation",
                "flying_speed",
                "countermeasure",
                "weather,day_time",
                "threat_range",
                "#threats"]

reqs = ["req_0", "req_1", "req_2", "req_3", "req_4", "req_5", "req_6", "req_7", "req_8", "req_9", "req_10", "req_11"]
reqsNamesInGraphs = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12"]

# read dataframe from csv
results = readFromCsv(pathToResults + 'results.csv')
resultsSHAP = readFromCsv(pathToResults + 'resultsSHAP.csv')
resultsPCA = readFromCsv(pathToResults + 'resultsPCA.csv')
resultsFI = readFromCsv(pathToResults + 'resultsFI.csv')
nReqs = len(results["nsga3_confidence"][0])
reqs = reqs[:nReqs]
reqsNamesInGraphs = reqsNamesInGraphs[:nReqs]
targetConfidence = np.full((1, nReqs), 0.8)[0]

if evaluate:
    evaluateAdaptations(results, resultsSHAP, resultsPCA, resultsFI, featureNames)

# read outcomes from csv
customOutcomes = pd.read_csv(pathToResults + 'customDataset.csv')
SHAPOutcomes = pd.read_csv(pathToResults + 'SHAPDataset.csv')
PCAOutcomes = pd.read_csv(pathToResults + 'PCADataset.csv')
FIOutcomes = pd.read_csv(pathToResults + 'FIDataset.csv')
nsga3Outcomes = pd.read_csv(pathToResults + 'nsga3Dataset.csv')

# build indices arrays
nsga3ConfidenceNames = ['nsga3_confidence_' + req for req in reqs]
nsga3OutcomeNames = ['nsga3_outcome_' + req for req in reqs]
customConfidenceNames = ['custom_confidence_' + req for req in reqs]
customOutcomeNames = ['custom_outcome_' + req for req in reqs]
SHAPcustomConfidenceNames = ['custom_confidence_' + req for req in reqs]
SHAPcustomOutcomeNames = ['custom_outcome_' + req for req in reqs]
PCAcustomConfidenceNames = ['custom_confidence_' + req for req in reqs]
PCAcustomOutcomeNames = ['custom_outcome_' + req for req in reqs]
FIcustomConfidenceNames = ['custom_confidence_' + req for req in reqs]
FIcustomOutcomeNames = ['custom_outcome_' + req for req in reqs]


#outcomes dataframe
outcomes = pd.concat([nsga3Outcomes[reqs], customOutcomes[reqs]], axis=1)
outcomes.columns = np.append(nsga3OutcomeNames, customOutcomeNames)
outcomes = outcomes[list(sum(zip(nsga3OutcomeNames, customOutcomeNames), ()))]
outcomesSHAP = pd.concat([nsga3Outcomes[reqs], SHAPOutcomes[reqs]], axis=1)
outcomesSHAP.columns = np.append(nsga3OutcomeNames, SHAPcustomOutcomeNames)
outcomesSHAP = outcomesSHAP[list(sum(zip(nsga3OutcomeNames, SHAPcustomOutcomeNames), ()))]
outcomesPCA = pd.concat([nsga3Outcomes[reqs], SHAPOutcomes[reqs]], axis=1)
outcomesPCA.columns = np.append(nsga3OutcomeNames, PCAcustomOutcomeNames)
outcomesPCA = outcomesPCA[list(sum(zip(nsga3OutcomeNames, PCAcustomOutcomeNames), ()))]
outcomesFI = pd.concat([nsga3Outcomes[reqs], FIOutcomes[reqs]], axis=1)
outcomesFI.columns = np.append(nsga3OutcomeNames, FIcustomOutcomeNames)
outcomesFI = outcomesFI[list(sum(zip(nsga3OutcomeNames, FIcustomOutcomeNames), ()))]


# decompose arrays columns into single values columns
nsga3Confidences = pd.DataFrame(results['nsga3_confidence'].to_list(),
                                columns=nsga3ConfidenceNames)
customConfidences = pd.DataFrame(results['custom_confidence'].to_list(),
                                 columns=customConfidenceNames)
customConfidencesSHAP = pd.DataFrame(results['custom_confidence'].to_list(),
                                 columns=SHAPcustomConfidenceNames)
customConfidencesPCA = pd.DataFrame(results['custom_confidence'].to_list(),
                                 columns=PCAcustomConfidenceNames)
customConfidencesFI = pd.DataFrame(results['custom_confidence'].to_list(),
                                 columns=FIcustomConfidenceNames)

# select sub-dataframes to plot
confidences = pd.concat([nsga3Confidences, customConfidences], axis=1)
confidences = confidences[list(sum(zip(nsga3Confidences.columns, customConfidences.columns), ()))]
scores = results[["nsga3_score", "custom_score"]]
times = results[["nsga3_time", "custom_time"]]
confidencesSHAP = pd.concat([nsga3Confidences, customConfidencesSHAP], axis=1)
confidencesSHAP = confidencesSHAP[list(sum(zip(nsga3Confidences.columns, customConfidencesSHAP.columns), ()))]
scoresSHAP = resultsSHAP[["nsga3_score", "custom_score"]]
timesSHAP = resultsSHAP[["nsga3_time", "custom_time"]]
confidencesPCA = pd.concat([nsga3Confidences, customConfidencesPCA], axis=1)
confidencesPCA = confidencesPCA[list(sum(zip(nsga3Confidences.columns, customConfidencesPCA.columns), ()))]
scoresPCA = resultsPCA[["nsga3_score", "custom_score"]]
timesPCA = resultsSHAP[["nsga3_time", "custom_time"]]
confidencesFI = pd.concat([nsga3Confidences, customConfidencesFI], axis=1)
confidencesFI = confidencesFI[list(sum(zip(nsga3Confidences.columns, customConfidencesFI.columns), ()))]
scoresFI = resultsSHAP[["nsga3_score", "custom_score"]]
timesFI = resultsSHAP[["nsga3_time", "custom_time"]]

# plots
plotPath = pathToResults + 'plots/'
if not os.path.exists(plotPath):
    os.makedirs(plotPath)

personalizedBoxPlot(confidences, "Confidences comparison", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scores, "Score comparison", path=plotPath)
personalizedBoxPlot(times, "Execution time comparison", path=plotPath, seconds=True, legendInside=True)
personalizedBoxPlot(confidencesSHAP, "Confidences comparison SHAP", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scoresSHAP, "Score comparison SHAP", path=plotPath)
personalizedBoxPlot(timesSHAP, "Execution time comparison SHAP", path=plotPath, seconds=True, legendInside=True)
personalizedBoxPlot(confidencesPCA, "Confidences comparison PCA", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scoresPCA, "Score comparison PCA", path=plotPath)
personalizedBoxPlot(timesPCA, "Execution time comparison PCA", path=plotPath, seconds=True, legendInside=True)
personalizedBoxPlot(confidencesFI, "Confidences comparison FI", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scoresFI, "Score comparison FI", path=plotPath)
personalizedBoxPlot(timesFI, "Execution time comparison FI", path=plotPath, seconds=True, legendInside=True)

# predicted successful adaptations
nsga3PredictedSuccessful = (confidences[nsga3ConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessful = (confidences[customConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulSHAP = (confidencesSHAP[SHAPcustomConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulPCA = (confidencesPCA[PCAcustomConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulFI = (confidencesFI[FIcustomConfidenceNames] > targetConfidence).all(axis=1)

personalizedBoxPlot(confidences[nsga3PredictedSuccessful], "Confidences comparison on NSGA-III predicted success", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scores[nsga3PredictedSuccessful], "Score comparison on NSGA-III predicted success", path=plotPath)
personalizedBoxPlot(times[nsga3PredictedSuccessful], "Execution time comparison on NSGA-III predicted success", path=plotPath, seconds=True, legendInside=True)
personalizedBoxPlot(confidencesSHAP[nsga3PredictedSuccessful], "SHAP confidences comparison on NSGA-III predicted success", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scoresSHAP[nsga3PredictedSuccessful], "SHAP score comparison on NSGA-III predicted success", path=plotPath)
personalizedBoxPlot(timesSHAP[nsga3PredictedSuccessful], "SHAP execution time comparison on NSGA-III predicted success", path=plotPath, seconds=True, legendInside=True)
personalizedBoxPlot(confidencesPCA[nsga3PredictedSuccessful], "PCA confidences comparison on NSGA-III predicted success", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scoresPCA[nsga3PredictedSuccessful], "PCA score comparison on NSGA-III predicted success", path=plotPath)
personalizedBoxPlot(timesPCA[nsga3PredictedSuccessful], "PCA execution time comparison on NSGA-III predicted success", path=plotPath, seconds=True, legendInside=True)
personalizedBoxPlot(confidencesFI[nsga3PredictedSuccessful], "FI confidences comparison on NSGA-III predicted success", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scoresFI[nsga3PredictedSuccessful], "FI score comparison on NSGA-III predicted success", path=plotPath)
personalizedBoxPlot(timesFI[nsga3PredictedSuccessful], "FI execution time comparison on NSGA-III predicted success", path=plotPath, seconds=True, legendInside=True)

print("NSGA-III predicted success rate: " + "{:.2%}".format(nsga3PredictedSuccessful.sum() / nsga3PredictedSuccessful.shape[0]))
print(str(nsga3Confidences.mean()) + "\n")
print("XDA predicted success rate:  " + "{:.2%}".format(customPredictedSuccessful.sum() / customPredictedSuccessful.shape[0]))
print(str(customConfidences.mean()) + "\n")
print("XDA SHAP predicted success rate:  " + "{:.2%}".format(customPredictedSuccessfulSHAP.sum() / customPredictedSuccessfulSHAP.shape[0]))
print(str(customConfidencesSHAP.mean()) + "\n")
print("XDA PCA predicted success rate:  " + "{:.2%}".format(customPredictedSuccessfulPCA.sum() / customPredictedSuccessfulPCA.shape[0]))
print(str(customConfidencesPCA.mean()) + "\n")
print("XDA FI predicted success rate:  " + "{:.2%}".format(customPredictedSuccessfulFI.sum() / customPredictedSuccessfulFI.shape[0]))
print(str(customConfidencesFI.mean()) + "\n")

print("NSGA-III mean probas of predicted success: \n" + str(nsga3Confidences[nsga3PredictedSuccessful].mean()) + '\n')
print("XDA mean probas of predicted success: \n" + str(customConfidences[customPredictedSuccessful].mean()) + '\n')
print("XDA SHAP mean probas of predicted success: \n" + str(customConfidencesSHAP[customPredictedSuccessfulSHAP].mean()) + '\n')
print("XDA PCA mean probas of predicted success: \n" + str(customConfidencesPCA[customPredictedSuccessfulPCA].mean()) + '\n')
print("XDA FI mean probas of predicted success: \n" + str(customConfidencesFI[customPredictedSuccessfulFI].mean()) + '\n')

# predicted successful adaptations
nsga3Successful = outcomes[nsga3OutcomeNames].all(axis=1)
customSuccessful = outcomes[customOutcomeNames].all(axis=1)
customSuccessfulSHAP = outcomesSHAP[SHAPcustomOutcomeNames].all(axis=1)
customSuccessfulPCA = outcomesPCA[PCAcustomOutcomeNames].all(axis=1)
customSuccessfulFI = outcomesFI[FIcustomOutcomeNames].all(axis=1)


nsga3SuccessRate = nsga3Successful.mean()
customSuccessRate = customSuccessful.mean()
customSuccessRateSHAP = customSuccessfulSHAP.mean()
customSuccessRatePCA = customSuccessfulPCA.mean()
customSuccessRateFI = customSuccessfulFI.mean()

# outcomes analysis
print("NSGA-III success rate: " + "{:.2%}".format(nsga3SuccessRate))
print(str(outcomes[nsga3OutcomeNames].mean()) + "\n")
print("XDA success rate:  " + "{:.2%}".format(customSuccessRate))
print(str(outcomes[customOutcomeNames].mean()) + "\n")
print("XDA SHAP success rate:  " + "{:.2%}".format(customSuccessRateSHAP))
print(str(outcomesSHAP[SHAPcustomOutcomeNames].mean()) + "\n")
print("XDA PCA success rate:  " + "{:.2%}".format(customSuccessRatePCA))
print(str(outcomesPCA[PCAcustomOutcomeNames].mean()) + "\n")
print("XDA FI success rate:  " + "{:.2%}".format(customSuccessRateFI))
print(str(outcomesFI[FIcustomOutcomeNames].mean()) + "\n")

successRateIndividual = pd.concat([outcomes[nsga3OutcomeNames].rename(columns=dict(zip(nsga3OutcomeNames, reqsNamesInGraphs))).mean(),
                                   outcomes[customOutcomeNames].rename(columns=dict(zip(customOutcomeNames, reqsNamesInGraphs))).mean()], axis=1)
successRateIndividual.columns = ['NSGA-III', 'XDA']
personalizedBarChart(successRateIndividual, "Success Rate Individual Reqs", plotPath)

successRateIndividualSHAP = pd.concat([outcomesSHAP[nsga3OutcomeNames].rename(columns=dict(zip(nsga3OutcomeNames, reqsNamesInGraphs))).mean(),
                                   outcomesSHAP[SHAPcustomOutcomeNames].rename(columns=dict(zip(SHAPcustomOutcomeNames, reqsNamesInGraphs))).mean()], axis=1)
successRateIndividualSHAP.columns = ['NSGA-III', 'XDA SHAP']
personalizedBarChart(successRateIndividualSHAP, "Success Rate Individual SHAP Reqs", plotPath)

successRateIndividualPCA = pd.concat([outcomesPCA[nsga3OutcomeNames].rename(columns=dict(zip(nsga3OutcomeNames, reqsNamesInGraphs))).mean(),
                                   outcomesPCA[PCAcustomOutcomeNames].rename(columns=dict(zip(PCAcustomOutcomeNames, reqsNamesInGraphs))).mean()], axis=1)
successRateIndividualPCA.columns = ['NSGA-III', 'XDA PCA']
personalizedBarChart(successRateIndividualPCA, "Success Rate Individual PCA Reqs", plotPath)

successRateIndividualFI = pd.concat([outcomesFI[nsga3OutcomeNames].rename(columns=dict(zip(nsga3OutcomeNames, reqsNamesInGraphs))).mean(),
                                   outcomesFI[customOutcomeNames].rename(columns=dict(zip(FIcustomOutcomeNames, reqsNamesInGraphs))).mean()], axis=1)
successRateIndividualFI.columns = ['NSGA-III', 'XDA FI']
personalizedBarChart(successRateIndividualFI, "Success Rate Individual FI Reqs", plotPath)



successRate = pd.DataFrame([[nsga3SuccessRate, customSuccessRate]], columns=["NSGA-III", "XDA"])
personalizedBarChart(successRate, "Success Rate", plotPath)

successRateOfPredictedSuccess = pd.DataFrame([[outcomes[nsga3OutcomeNames][nsga3PredictedSuccessful].all(axis=1).mean(),
                                               outcomes[customOutcomeNames][customPredictedSuccessful].all(axis=1).mean()]],
                                             columns=["NSGA-III", "XDA"])
personalizedBarChart(successRateOfPredictedSuccess, "Success Rate of Predicted Success", plotPath)

successRateSHAP = pd.DataFrame([[nsga3SuccessRate, customSuccessRateSHAP]], columns=["NSGA-III", "XDA SHAP"])
personalizedBarChart(successRateSHAP, "Success Rate SHAP", plotPath)

successRateOfPredictedSuccessSHAP = pd.DataFrame([[outcomesSHAP[nsga3OutcomeNames][nsga3PredictedSuccessful].all(axis=1).mean(),
                                               outcomesSHAP[SHAPcustomOutcomeNames][customPredictedSuccessfulSHAP].all(axis=1).mean()]],
                                             columns=["NSGA-III", "XDA SHAP"])
personalizedBarChart(successRateOfPredictedSuccessSHAP, "Success Rate of Predicted Success SHAP", plotPath)

successRatePCA = pd.DataFrame([[nsga3SuccessRate, customSuccessRatePCA]], columns=["NSGA-III", "XDA PCA"])
personalizedBarChart(successRatePCA, "Success Rate PCA", plotPath)

successRateOfPredictedSuccessPCA = pd.DataFrame([[outcomesPCA[nsga3OutcomeNames][nsga3PredictedSuccessful].all(axis=1).mean(),
                                               outcomesPCA[PCAcustomOutcomeNames][customPredictedSuccessfulPCA].all(axis=1).mean()]],
                                             columns=["NSGA-III", "XDA PCA"])
personalizedBarChart(successRateOfPredictedSuccessPCA, "Success Rate of Predicted Success PCA", plotPath)

successRateFI = pd.DataFrame([[nsga3SuccessRate, customSuccessRateFI]], columns=["NSGA-III", "XDA FI"])
personalizedBarChart(successRateFI, "Success Rate FI", plotPath)

successRateOfPredictedSuccessFI = pd.DataFrame([[outcomesFI[nsga3OutcomeNames][nsga3PredictedSuccessful].all(axis=1).mean(),
                                               outcomesFI[FIcustomOutcomeNames][customPredictedSuccessfulFI].all(axis=1).mean()]],
                                             columns=["NSGA-III", "XDA FI"])
personalizedBarChart(successRateOfPredictedSuccessFI, "Success Rate of Predicted Success FI", plotPath)

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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def personalizedBoxPlot(data, name, columnNames=None, percentage=False, path=None, show=False, seconds=False,
                        legendInside=False):


    # Impostazioni per le dimensioni dell'immagine
    fig = plt.figure(figsize=(15, 8))  # 1500x800

    ax1 = fig.add_subplot(111)
    bp = ax1.boxplot(data, patch_artist=True, notch=True, vert=True)

    # Definizione dei colori per gli algoritmi
    algorithm_colors = [
        '#FF5733',
        '#6B8E23',
        '#4169E1',
        '#FFD700',
        '#8A2BE2'
    ]

    for i, box in enumerate(bp['boxes']):
        # Trova il nome dell'algoritmo associato alla colonna attuale
        algorithm = i % 5
        box.set_facecolor(algorithm_colors[algorithm])

    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B', linewidth=1.5, linestyle=":")

    for cap in bp['caps']:
        cap.set(color='#8B008B', linewidth=2)

    for median in bp['medians']:
        median.set(color='red', linewidth=3)

    for flier in bp['fliers']:
        flier.set(marker='D', color='#e7298a', alpha=0.5)

    if columnNames is not None and len(columnNames) > 1:
        ax1.xaxis.set_ticks(np.arange(1.5, len(columnNames) * 2, step=2), columnNames)
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

    ax1.set_yscale('log')  # Imposta la scala logaritmica sull'asse y

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    if legendInside:
        ax1.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2], bp["boxes"][3], bp["boxes"][4]],
                   ["NSGA-III", "XDA", "XDA SHAP", "XDA PCA", "XDA FI"])
    else:
        ax1.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2], bp["boxes"][3], bp["boxes"][4]],
                   ["NSGA-III", "XDA", "XDA SHAP", "XDA PCA", "XDA FI"],
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


def personalizedBarChart(data, name, path=None, show=False, percentage=False):

    colors = plt.cm.Spectral(np.linspace(0, 1, 5))

    ax = data.plot.bar(title=name, color=colors, figsize=(15, 8))


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
confidences_concat = pd.concat([nsga3Confidences, customConfidences, customConfidencesSHAP, customConfidencesPCA, customConfidencesFI], axis=1)
confidences_concat = confidences[list(sum(zip(nsga3Confidences.columns, customConfidences.columns, customConfidencesSHAP.columns, customConfidencesPCA.columns, customConfidencesFI.columns), ()))]
scores = pd.concat([results["nsga3_score"], results["custom_score"], resultsSHAP["custom_score"],
                    resultsPCA["custom_score"], resultsFI["custom_score"]], axis=1)
times = pd.concat([results["nsga3_time"], results["custom_time"],
                            resultsSHAP["custom_time"],
                            resultsPCA["custom_time"],
                            resultsFI["custom_time"]], axis=1)
confidencesSHAP = pd.concat([customConfidencesSHAP], axis=1)
confidencesSHAP = confidencesSHAP[list(sum(zip(customConfidencesSHAP.columns), ()))]
scoresSHAP = resultsSHAP[["custom_score"]]
timesSHAP = resultsSHAP[["custom_time"]]
confidencesPCA = pd.concat([customConfidencesPCA], axis=1)
confidencesPCA = confidencesPCA[list(sum(zip(customConfidencesPCA.columns), ()))]
scoresPCA = resultsPCA[["custom_score"]]
timesPCA = resultsSHAP[["custom_time"]]
confidencesFI = pd.concat([customConfidencesFI], axis=1)
confidencesFI = confidencesFI[list(sum(zip(customConfidencesFI.columns), ()))]
scoresFI = resultsSHAP[["custom_score"]]
timesFI = resultsSHAP[["custom_time"]]

# plots
plotPath = pathToResults + 'plots/'
if not os.path.exists(plotPath):
    os.makedirs(plotPath)


personalizedBoxPlot(confidences_concat, "Confidences comparison", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scores, "Score comparison", path=plotPath)
personalizedBoxPlot(times, "Execution time comparison", path=plotPath, seconds=True, legendInside=True)

# predicted successful adaptations
nsga3PredictedSuccessful = (confidences[nsga3ConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessful = (confidences[customConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulSHAP = (confidencesSHAP[SHAPcustomConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulPCA = (confidencesPCA[PCAcustomConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulFI = (confidencesFI[FIcustomConfidenceNames] > targetConfidence).all(axis=1)
predicted_successful_combined = pd.DataFrame({
    'nsga3PredictedSuccessful': nsga3PredictedSuccessful,
    'customPredictedSuccessful': customPredictedSuccessful,
    'customPredictedSuccessfulSHAP': customPredictedSuccessfulSHAP,
    'customPredictedSuccessfulPCA': customPredictedSuccessfulPCA,
    'customPredictedSuccessfulFI': customPredictedSuccessfulFI
})

personalizedBoxPlot(confidences_concat[nsga3PredictedSuccessful], "Confidences comparison on NSGA-III predicted success", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scores[nsga3PredictedSuccessful], "Score comparison on NSGA-III predicted success", path=plotPath)
personalizedBoxPlot(times[nsga3PredictedSuccessful], "Execution time comparison on NSGA-III predicted success", path=plotPath, seconds=True, legendInside=True)

print("NSGA-III predicted success rate: " + "{:.2%}".format(
    nsga3PredictedSuccessful.sum() / nsga3PredictedSuccessful.shape[0]))
print(str(nsga3Confidences.mean()) + "\n")
print("XDA predicted success rate:  " + "{:.2%}".format(
    customPredictedSuccessful.sum() / customPredictedSuccessful.shape[0]))
print(str(customConfidences.mean()) + "\n")
print("XDA SHAP predicted success rate:  " + "{:.2%}".format(
    customPredictedSuccessfulSHAP.sum() / customPredictedSuccessfulSHAP.shape[0]))
print(str(customConfidencesSHAP.mean()) + "\n")
print("XDA PCA predicted success rate:  " + "{:.2%}".format(
    customPredictedSuccessfulPCA.sum() / customPredictedSuccessfulPCA.shape[0]))
print(str(customConfidencesPCA.mean()) + "\n")
print("XDA FI predicted success rate:  " + "{:.2%}".format(
    customPredictedSuccessfulFI.sum() / customPredictedSuccessfulFI.shape[0]))
print(str(customConfidencesFI.mean()) + "\n")

print("NSGA-III mean probas of predicted success: \n" + str(nsga3Confidences[nsga3PredictedSuccessful].mean()) + '\n')
print("XDA mean probas of predicted success: \n" + str(customConfidences[customPredictedSuccessful].mean()) + '\n')
print("XDA SHAP mean probas of predicted success: \n" + str(
    customConfidencesSHAP[customPredictedSuccessfulSHAP].mean()) + '\n')
print("XDA PCA mean probas of predicted success: \n" + str(
    customConfidencesPCA[customPredictedSuccessfulPCA].mean()) + '\n')
print(
    "XDA FI mean probas of predicted success: \n" + str(customConfidencesFI[customPredictedSuccessfulFI].mean()) + '\n')

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
                                   outcomes[customOutcomeNames].rename(columns=dict(zip(customOutcomeNames, reqsNamesInGraphs))).mean(),
                                   outcomesSHAP[SHAPcustomOutcomeNames].rename(columns=dict(zip(SHAPcustomOutcomeNames, reqsNamesInGraphs))).mean(),
                                  outcomesPCA[PCAcustomOutcomeNames].rename(columns=dict(zip(PCAcustomOutcomeNames, reqsNamesInGraphs))).mean(),
                                   outcomesFI[customOutcomeNames].rename(columns=dict(zip(FIcustomOutcomeNames, reqsNamesInGraphs))).mean()],
                                  axis=1)
successRateIndividual.columns = ['NSGA-III', 'XDA', 'XDA SHAP', 'XDA PCA', 'XDA FI']
personalizedBarChart(successRateIndividual, "Success Rate Individual Reqs", plotPath)

successRate = pd.DataFrame([[nsga3SuccessRate, customSuccessRate, customSuccessRateSHAP, customSuccessRatePCA, customSuccessRateFI]],
                           columns=["NSGA-III", "XDA", "XDA SHAP", "XDA PCA", "XDA FI"])
personalizedBarChart(successRate, "Success Rate", plotPath)

successRateOfPredictedSuccess = pd.DataFrame([[outcomes[nsga3OutcomeNames][nsga3PredictedSuccessful].all(axis=1).mean(),
                                               outcomes[customOutcomeNames][customPredictedSuccessful].all(axis=1).mean(),
                                               outcomesSHAP[SHAPcustomOutcomeNames][customPredictedSuccessfulSHAP].all(axis=1).mean(),
                                               outcomesPCA[PCAcustomOutcomeNames][customPredictedSuccessfulPCA].all(axis=1).mean(),
                                               outcomesFI[FIcustomOutcomeNames][customPredictedSuccessfulFI].all(axis=1).mean()]],
                                             columns=["NSGA-III", "XDA", "XDA SHAP", "XDA PCA", "XDA FI"])
personalizedBarChart(successRateOfPredictedSuccess, "Success Rate of Predicted Success", plotPath)


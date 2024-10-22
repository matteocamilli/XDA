import csv
import os
import sys
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from util import readFromCsv, evaluateAdaptations

font = {'family': 'sans',
        'weight': 'normal',
        'size': 12}

matplotlib.rc('font', **font)


def rankingPerf(df, path=None, legendInside=False, time=False, name ='FI'):

    model_names = ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier',
                   'GradientBoostingClassifier', 'NeuralNetwork']
    if time:
        metric = 'Time'
        metric_label = "Time (s)"
    else:
        metric = 'Memory Peak (MB)'
        metric_label = "Peak Memory (MB)"

    boxplot_data = []
    for model in model_names:
        boxplot_data.append(df[df['Model'] == model][metric].values)

    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(111)

    bp = ax1.boxplot(boxplot_data, patch_artist=True, notch=True, vert=True)

    colors = ['#FF5733', '#6B8E23', '#1E90FF', '#FFD700', '#FF69B4']

    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B', linewidth=1.5, linestyle=":")

    for cap in bp['caps']:
        cap.set(color='#8B008B', linewidth=2)

    for median in bp['medians']:
        median.set(color='black', linewidth=3)

    for flier in bp['fliers']:
        flier.set(marker='D', color='#e7298a', alpha=0.5)

    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % len(colors)])

    for i in range(1, len(model_names)):
        ax1.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1.5)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    legend_labels = model_names
    ax1.set_yscale('log')

    ax1.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors[:len(model_names)]],
                legend_labels,
                ncol=2, loc='best', bbox_to_anchor=(0.5, -0.1))

    if time:
        if name == 'FI':
            plt.title("Model Execution Time FI")
        else:
            plt.title("Model Execution Time SHAP")
    else:
        if name == 'FI':
            plt.title("Memory Peak (MB) FI")
        else:
            plt.title("Memory Peak (MB) SHAP")

    plt.ylabel(metric_label)

    if path is not None:
        if time:
            if name == 'FI':
                plt.savefig(path + 'Model_Performance_Time_FI.png')
            else:
                plt.savefig(path + 'Model_Performance_Time_SHAP.png')
        else:
            if name == 'FI':
                plt.savefig(path + "Model_Performance_Memory_FI.png")
            else:
                plt.savefig(path + "Model_Performance_Memory_SHAP.png")

    #plt.show()


def pdp_plot(df, path, legendInside=False, time=False):
    dataset_names = ['Robot', 'RobotDouble', 'UAV', 'UAVDouble', 'Drive', 'DriveDouble']
    values_per_dataset = 20

    if time:
        pdp = df['PDP_Time']
        spdp = df['SPDP_Time']
        metric = "Iime (s)"
    else:
        pdp = df['PDP_Peak_Memory_MB']
        spdp = df['SPDP_Peak_Memory_MB']
        metric = "Peak Memory (MB)"
    pdp_segments = []
    spdp_segments = []

    for i in range(0, len(pdp), values_per_dataset):
        pdp_segments.append(pdp[i:i + values_per_dataset].values)
        spdp_segments.append(spdp[i:i + values_per_dataset].values)

    boxplot_data = []
    for pdp_seg, spdp_seg in zip(pdp_segments, spdp_segments):
        boxplot_data.append(pdp_seg)
        boxplot_data.append(spdp_seg)

    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(111)

    bp = ax1.boxplot(boxplot_data, patch_artist=True, notch=True, vert=True)

    colors = ['#FF5733', '#6B8E23']

    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B', linewidth=1.5, linestyle=":")

    for cap in bp['caps']:
        cap.set(color='#8B008B', linewidth=2)

    for median in bp['medians']:
        median.set(color='black', linewidth=3)

    for flier in bp['fliers']:
        flier.set(marker='D', color='#e7298a', alpha=0.5)

    for i, box in enumerate(bp['boxes']):
        group_index = i % 2
        box.set_facecolor(colors[group_index])

    x_labels = []
    for dataset in dataset_names:
        x_labels.append(f'{dataset} PDP')
        x_labels.append(f'{dataset} SPDP')

    for i in range(1, len(dataset_names)):
        ax1.axvline(x=2 * i + 0.5, color='gray', linestyle='--', linewidth=1.5)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    legend_labels = ["PDP", "SPDP"]
    plt.ylabel(metric)

    ax1.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors[:2]],
                legend_labels,
                ncol=2, loc='best', bbox_to_anchor=(0.5, -0.1))

    if time:
        plt.title("PDP and SPDP time execution")
    else:
        plt.title("PDP and SPDP peak memory (MB)")

    if path is not None:
        if time:
            plt.savefig(path + 'PDP and SPSD time')
        else:
            plt.savefig(path + "PDP and SPDP peak memory (MB)")

    #plt.show()


def memoryPlot(data, path):
    dataMemory = {
        'CustomMemory': data['CustomMemory'],
        'SHAPMemory': data['SHAPMemory'],
        'FIMemory': data['FIMemory'],
        'FitestMemory': data['FitestMemory'],
        'RandomMemory': data['RandomMemory'],
        'NSGA3Memory': data['NSGA3Memory']
    }

    plt.figure(figsize=(15, 8))
    for key in dataMemory:
        plt.plot(dataMemory[key], label=key)

    plt.xlabel('Test Number')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Tests')
    plt.yscale('log')
    plt.legend()
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(path + 'Memory.png')
    plt.close()

    personalizedBoxPlotUnified(data, "Memory Box Plot", path=plotPath, log=True)


def personalizedBoxPlotUnified(data, name, nReq=4, columnNames=None, percentage=False, path=None, show=False,
                               seconds=False, legendInside=False, numAlgorithms=6, confidence=False, log=False):
    fig = plt.figure(figsize=(20, 10))  # 1500x800

    ax1 = fig.add_subplot(111)
    bp = ax1.boxplot(data, patch_artist=True, notch=True, vert=True)

    algorithm_colors = [
        '#FF5733',  # Arancione acceso
        '#6B8E23',  # Verde oliva
        '#4169E1',  # Blu reale
        '#FF00FF',  # Magenta
        '#FFD700',  # Oro
        '#00CED1',  # Turchese scuro (usato solo se ci sono 6 dataset)
    ]

    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B', linewidth=1.5, linestyle=":")

    for cap in bp['caps']:
        cap.set(color='#8B008B', linewidth=2)

    for median in bp['medians']:
        median.set(color='black', linewidth=3)

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

    if log:
        ax1.set_yscale('log')
    for i, box in enumerate(bp['boxes']):
        group_index = i % numAlgorithms
        box.set_facecolor(algorithm_colors[group_index])

    if confidence:
        for i in range(1, nReq + 1):
            plt.axvline(x=numAlgorithms * i + 0.5, color='gray', linestyle='--', linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])

    legend_labels = ["XDA", "XDA SHAP", "XDA FI", "Fitest", "Random"]
    if numAlgorithms == 6:
        legend_labels.append("NSGA-III")
    else:
        legend_labels.append("NSGA-III (data not available)")

    if legendInside:
        ax1.legend([plt.Line2D([0], [0], color=color, lw=4) for color in algorithm_colors[:numAlgorithms]],
                   legend_labels)
    else:
        ax1.legend([plt.Line2D([0], [0], color=color, lw=4) for color in algorithm_colors[:numAlgorithms]],
                   legend_labels,
                   ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))

    plt.title(name)

    if path is not None:
        plt.savefig(path + name)

    if show:
        plt.show()
    else:
        plt.clf()


def personalizedBarChart(data, name, nReq, path=None, show=False, percentage=False):
    colors = plt.cm.Spectral(np.linspace(0, 1, 6))

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

    ax.legend(loc='best')

    if path is not None:
        plt.savefig(path + name)

    if show:
        plt.show()
    else:
        plt.clf()


os.chdir(sys.path[0])
evaluate = False

pathToResults = ("../results/uavAllv3/")

featureNames = ['formation', 'flying_speed', 'countermeasure', 'weather', 'day_time', 'threat_range', '#threats']
#featureNames = ['cruise speed','image resolution','illuminance','controls responsiveness','power','smoke intensity','obstacle size','obstacle distance','firm obstacle']
# featureNames = ['car_speed', 'p_x', 'p_y', 'orientation', 'weather', 'road_shape']

reqs = ["req_0", "req_1", "req_2", "req_3", "req_4", "req_5", "req_6", "req_7", "req_8", "req_9", "req_10", "req_11"]
reqsNamesInGraphs = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12"]

# read dataframe from csv

memoryPdp = pd.read_csv('../results/profiling/memory_usage_pdp_peak.csv')
memoryPdp = pd.DataFrame(memoryPdp)
timePdp = pd.read_csv('../results/profiling/time_pdp_spdp.csv')
profileFI = pd.read_csv('../results/profiling/FI_profile.csv')
profileFI = pd.DataFrame(profileFI)
profileSHAP = pd.read_csv('../results/profiling/SHAP_profile.csv')
profileSHAP = pd.DataFrame(profileSHAP)
results = readFromCsv(pathToResults + 'results.csv')
resultsSHAP = readFromCsv(pathToResults + 'resultsSHAP.csv')
resultsFI = readFromCsv(pathToResults + 'resultsFI.csv')
resultsFitest = readFromCsv(pathToResults + 'resultsFitest.csv')
resultsRandom = readFromCsv(pathToResults + 'resultsRandom.csv')
resultsNSGA = readFromCsv(pathToResults + 'resultsNSGA.csv')
resultMemory = pd.read_csv(pathToResults + 'memory_results.csv')
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
FitestOutcomes = pd.read_csv(pathToResults + 'FitestDataset.csv')
nsga3Outcomes = pd.read_csv(pathToResults + 'NSGADataset.csv')
RandomOutcomes = pd.read_csv(pathToResults + 'RandomDataset.csv')

# build indices arrays
nsga3ConfidenceNames = ['custom_confidence_' + req for req in reqs]
nsga3OutcomeNames = ['custom_outcome_' + req for req in reqs]
customConfidenceNames = ['custom_confidence_' + req for req in reqs]
customOutcomeNames = ['custom_outcome_' + req for req in reqs]
SHAPcustomConfidenceNames = ['custom_confidence_' + req for req in reqs]
SHAPcustomOutcomeNames = ['custom_outcome_' + req for req in reqs]
FIcustomConfidenceNames = ['custom_confidence_' + req for req in reqs]
FIcustomOutcomeNames = ['custom_outcome_' + req for req in reqs]
FitestcustomConfidenceNames = ['custom_confidence_' + req for req in reqs]
FitestcustomOutcomeNames = ['custom_outcome_' + req for req in reqs]
RandomcustomOutcomesNames = ['custom_outcome_' + req for req in reqs]
RandomcustomConfidenceNames = ['custom_confidence_' + req for req in reqs]

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
outcomesFitest = FitestOutcomes[reqs]
outcomesFitest.columns = np.array(FitestcustomOutcomeNames)
outcomesFitest = outcomesFitest[list(sum(zip(FitestcustomOutcomeNames), ()))]
outcomesRandom = RandomOutcomes[reqs]
outcomesRandom.columns = np.array(RandomcustomOutcomesNames)
outcomesRandom = outcomesRandom[list(sum(zip(RandomcustomOutcomesNames), ()))]
outcomesNSGA = nsga3Outcomes[reqs]
outcomesNSGA.columns = np.array(nsga3OutcomeNames)
outcomesNSGA = outcomesNSGA[list(sum(zip(nsga3OutcomeNames), ()))]
# decompose arrays columns into single values columns
#nsga3Confidences = pd.DataFrame(results['nsga3_confidence'].to_list(),
#                               columns=nsga3ConfidenceNames)
customConfidences = pd.DataFrame(results['custom_confidence'].to_list(),
                                 columns=customConfidenceNames)
customConfidencesSHAP = pd.DataFrame(resultsSHAP['custom_confidence'].to_list(),
                                     columns=SHAPcustomConfidenceNames)
customConfidencesFI = pd.DataFrame(resultsFI['custom_confidence'].to_list(),
                                   columns=FIcustomConfidenceNames)
customConfidencesFitest = pd.DataFrame(resultsFitest['custom_confidence'].to_list(),
                                       columns=FitestcustomConfidenceNames)
customConfidencesRandom = pd.DataFrame(resultsRandom['custom_confidence'].to_list(),
                                       columns=RandomcustomConfidenceNames)
NSGAConfidences = pd.DataFrame(resultsNSGA['custom_confidence'].to_list(),
                               columns=nsga3ConfidenceNames)

# select sub-dataframes to plot
confidences = customConfidences
confidences = confidences[list(sum(zip(customConfidences.columns), ()))]
confidences_concat = pd.concat(
    [customConfidences, customConfidencesSHAP, customConfidencesFI, customConfidencesFitest, customConfidencesRandom,
     NSGAConfidences], axis=1)

num_columns = confidences_concat.shape[1]
order = []
for i in range(len(reqs)):
    order += list(range(i, num_columns, len(reqs)))

confidences_reordered = confidences_concat.iloc[:, order]

scores = pd.concat([results["custom_score"], resultsSHAP["custom_score"],
                    resultsFI["custom_score"], resultsFitest["custom_score"], resultsRandom["custom_score"],
                    resultsNSGA["custom_score"]], axis=1)
times = pd.concat([results["custom_time"],
                   resultsSHAP["custom_time"],
                   resultsFI["custom_time"], resultsFitest["custom_time"], resultsRandom["custom_time"],
                   resultsNSGA["custom_time"]], axis=1)
confidencesSHAP = pd.concat([customConfidencesSHAP], axis=1)
confidencesSHAP = confidencesSHAP[list(sum(zip(customConfidencesSHAP.columns), ()))]
scoresSHAP = resultsSHAP[["custom_score"]]
timesSHAP = resultsSHAP[["custom_time"]]
confidencesFI = pd.concat([customConfidencesFI], axis=1)
confidencesFI = confidencesFI[list(sum(zip(customConfidencesFI.columns), ()))]
scoresFI = resultsFI[["custom_score"]]
timesFI = resultsFI[["custom_time"]]
confidencesFitest = pd.concat([customConfidencesFitest], axis=1)
confidencesFitest = confidencesFitest[list(sum(zip(customConfidencesFitest.columns), ()))]
scoresFitest = resultsFitest[["custom_score"]]
timesFitest = resultsFitest[["custom_time"]]
confidencesRandom = pd.concat([customConfidencesRandom], axis=1)
confidencesRandom = confidencesRandom[list(sum(zip(customConfidencesRandom.columns), ()))]
scoresRandom = resultsRandom[["custom_score"]]
timesRandom = resultsRandom[["custom_time"]]
confidencesNSGA = pd.concat([NSGAConfidences], axis=1)
confidencesNSGA = confidencesNSGA[list(sum(zip(NSGAConfidences.columns), ()))]
scoresNSGA = resultsNSGA[["custom_score"]]
timesNSGA = resultsNSGA[["custom_time"]]

# plots

plotPath = pathToResults + 'plots/'

if not os.path.exists(plotPath):
    os.makedirs(plotPath)

personalizedBoxPlotUnified(confidences_reordered, "Confidences comparison", nReqs, reqsNamesInGraphs, path=plotPath,
                           percentage=False, confidence=True)
personalizedBoxPlotUnified(scores, "Score comparison", path=plotPath)
personalizedBoxPlotUnified(times, "Execution time comparison", path=plotPath, seconds=True, legendInside=True, log=True)

# predicted successful adaptations
#nsga3PredictedSuccessful = (confidences[nsga3ConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessful = (confidences[customConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulSHAP = (confidencesSHAP[SHAPcustomConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulFI = (confidencesFI[FIcustomConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulFitest = (confidencesFitest[FitestcustomConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessfulRandom = (confidencesRandom[RandomcustomConfidenceNames] > targetConfidence).all(axis=1)
nsga3PredictedSuccessful = (confidencesNSGA[nsga3ConfidenceNames] > targetConfidence).all(axis=1)
predicted_successful_combined = pd.DataFrame({
    'customPredictedSuccessful': customPredictedSuccessful,
    'customPredictedSuccessfulSHAP': customPredictedSuccessfulSHAP,
    'customPredictedSuccessfulFI': customPredictedSuccessfulFI,
    'customPredictedSuccessfulFitest': customPredictedSuccessfulFitest,
    'customPredictedSuccessfulRandom': customPredictedSuccessfulRandom,
    'nsga3PredictedSuccessful': nsga3PredictedSuccessful
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
nsga3Successful = outcomesNSGA[nsga3OutcomeNames].all(axis=1)
customSuccessful = outcomes[customOutcomeNames].all(axis=1)
customSuccessfulSHAP = outcomesSHAP[customOutcomeNames].all(axis=1)
customSuccessfulFI = outcomesFI[customOutcomeNames].all(axis=1)
customSuccessfulFitest = outcomesFitest[customOutcomeNames].all(axis=1)
customSuccessfulRandom = outcomesRandom[customOutcomeNames].all(axis=1)

nsga3SuccessRate = nsga3Successful.mean()
customSuccessRate = customSuccessful.mean()
customSuccessRateSHAP = customSuccessfulSHAP.mean()
customSuccessRateFI = customSuccessfulFI.mean()
customSuccessRateFitest = customSuccessfulFitest.mean()
customSuccessRateRandom = customSuccessfulRandom.mean()

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
     outcomesFI[customOutcomeNames].rename(columns=dict(zip(FIcustomOutcomeNames, reqsNamesInGraphs))).mean(),
     outcomesFitest[customOutcomeNames].rename(columns=dict(zip(FitestcustomOutcomeNames, reqsNamesInGraphs))).mean(),
     outcomesRandom[customOutcomeNames].rename(columns=dict(zip(RandomcustomOutcomesNames, reqsNamesInGraphs))).mean(),
     outcomesNSGA[customOutcomeNames].rename(columns=dict(zip(nsga3OutcomeNames, reqsNamesInGraphs))).mean()],
    axis=1)
successRateIndividual.columns = ['XDA', 'XDA SHAP', 'XDA FI', 'Fitest', 'Random', 'NSGA3']
personalizedBarChart(successRateIndividual, "Success Rate Individual Reqs", nReqs, plotPath)

successRate = pd.DataFrame(
    [[customSuccessRate, customSuccessRateSHAP, customSuccessRateFI, customSuccessRateFitest, customSuccessRateRandom,
      nsga3SuccessRate]],
    columns=["XDA", "XDA SHAP", "XDA FI", "Fitest", "Random", "NSGA3"])
personalizedBarChart(successRate, "Success Rate", nReqs, plotPath)

successRateOfPredictedSuccess = pd.DataFrame([[outcomes[customOutcomeNames][customPredictedSuccessful].all(
    axis=1).mean(),
                                               outcomesSHAP[SHAPcustomOutcomeNames][customPredictedSuccessfulSHAP].all(
                                                   axis=1).mean(),
                                               outcomesFI[FIcustomOutcomeNames][customPredictedSuccessfulFI].all(
                                                   axis=1).mean(),
                                               outcomesFitest[FitestcustomOutcomeNames][
                                                   customPredictedSuccessfulFitest].all(
                                                   axis=1).mean(),
                                               outcomesRandom[RandomcustomOutcomesNames][
                                                   customPredictedSuccessfulRandom].all(
                                                   axis=1).mean(),
                                               outcomesNSGA[nsga3OutcomeNames][nsga3PredictedSuccessful].all(
                                                   axis=1).mean()
                                               ]],
                                             columns=["XDA", "XDA SHAP", "XDA FI", "Fitest", "Random", "NSGA3"])
personalizedBarChart(successRateOfPredictedSuccess, "Success Rate of Predicted Success", nReqs, plotPath)

memoryPlot(resultMemory, plotPath)
pdp_plot(timePdp, '../results/', time=True)
pdp_plot(memoryPdp, '../results/')
rankingPerf(profileFI, '../results/', time=True)
rankingPerf(profileFI, '../results/')
rankingPerf(profileSHAP, '../results/', time=True, name="SHAP")
rankingPerf(profileSHAP, '../results/', name="SHAP")

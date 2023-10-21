import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import readFromCsv, evaluateAdaptations


def personalizedBoxPlot(data, name, rotation=0):
    columns = data.columns
    nColumns = len(columns)
    fig = plt.figure(figsize=(10, 10 * nColumns/2))
    ax1 = fig.add_subplot(nColumns, 1, 1)

    # Creating axes instance
    bp = ax1.boxplot(data, patch_artist=True,
                     notch='True', vert=True)

    colors = plt.cm.hsv(np.linspace(.1, .9, nColumns))
    #colors = np.append(colors[0::2], colors[1::2], axis=0)

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
    ax1.set_xticklabels(data.columns, rotation = rotation)

    # Adding title
    plt.title(name)

    # Removing top axes and right axes
    # ticks
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    for i in range(int(nColumns/2)):
        i2 = i + int(nColumns/2)
        axn = fig.add_subplot(nColumns, 1, i + 2)
        subset = data[[columns[i], columns[i2]]]
        subset = subset.sort_values(columns[i2])
        subset = subset.reset_index(drop=True)
        # axn.title.set_text(columns[i] + ' | ' + columns[i + int(nColumns/2)])
        subset.plot(ax=axn, color=colors[[i, i2]])

    fig.show()


os.chdir(sys.path[0])
evaluate = False

pathToResults = '../results/1ss/allReqs/'

featureNames = ["cruise speed",
                    "image resolution",
                    "illuminance",
                    "controls responsiveness",
                    "power",
                    "smoke intensity",
                    "obstacle size",
                    "obstacle distance",
                    "firm obstacle"]

reqs = ["req_0", "req_1", "req_2", "req_3"]

# read dataframe from csv
results = readFromCsv(pathToResults + 'results.csv')
nReqs = len(results["nsga3_confidence"][0])
reqs = reqs[:nReqs]
targetConfidence = np.full((1, nReqs), 0.8)[0]

if evaluate:
    evaluateAdaptations(results, featureNames)

#read outcomes from csv
customDataset = pd.read_csv(pathToResults + 'customDataset.csv')
nsga3Dataset = pd.read_csv(pathToResults + 'nsga3Dataset.csv')

#build indices arrays
nsga3ConfidenceNames = ['nsga3_confidence_' + req for req in reqs]
nsga3OutcomeNames = ['nsga3_outcome_' + req for req in reqs]
customConfidenceNames = ['custom_confidence_' + req for req in reqs]
customOutcomeNames = ['custom_confidence_' + req for req in reqs]

#outcomes dataframe
outcomes = pd.concat([nsga3Dataset[reqs].mean(axis=1), customDataset[reqs].mean(axis=1)], axis=1)
outcomes.columns = ['nsga3', 'custom']

# decompose arrays columns into single values columns
nsga3Confidences = pd.DataFrame(results['nsga3_confidence'].to_list(),
                                columns=nsga3ConfidenceNames)
customConfidences = pd.DataFrame(results['custom_confidence'].to_list(),
                                 columns=customConfidenceNames)

# select sub-dataframes to plot
confidences = pd.concat([nsga3Confidences, customConfidences], axis=1)
scores = results[["nsga3_score", "custom_score"]]
times = results[["nsga3_time", "custom_time"]]

#plots
personalizedBoxPlot(confidences, "Confidences comparison", 30)
personalizedBoxPlot(scores, "Score comparison")
personalizedBoxPlot(times, "Execution time comparison")

#mapping
bothSuccesful = pd.concat([confidences[nsga3ConfidenceNames] > targetConfidence, confidences[customConfidenceNames] > targetConfidence], axis=1).all(axis=1)
onlyNsga3Succesful = pd.concat([confidences[nsga3ConfidenceNames] > targetConfidence, (confidences[customConfidenceNames] <= targetConfidence).any(axis=1)], axis=1).all(axis=1)
onlyCustomSuccesful = pd.concat([(confidences[nsga3ConfidenceNames] <= targetConfidence).any(axis=1), confidences[customConfidenceNames] > targetConfidence], axis=1).all(axis=1)
noneSuccesful = pd.concat([(confidences[nsga3ConfidenceNames] <= targetConfidence).any(axis=1), (confidences[customConfidenceNames] <= targetConfidence).any(axis=1)], axis=1).all(axis=1)

#results
averages = pd.concat([outcomes[bothSuccesful].mean(),
                      outcomes[onlyNsga3Succesful].mean(),
                      outcomes[onlyCustomSuccesful].mean(),
                      outcomes[noneSuccesful].mean()], axis=1)
averages.columns = ['both', 'nsga3_only', 'custom_only', 'none']

print(str(averages) + "\n")

customAverages = confidences[customConfidenceNames]
nsga3Averages = confidences[nsga3ConfidenceNames]

print("nsga3 mean:  " + str(nsga3Averages.mean(axis=1).mean()))
print("nsga3:  " + str(nsga3Averages.mean()) + "\n")
print("custom mean:  " + str(customAverages.mean(axis=1).mean()))
print("custom: " + str(customAverages.mean()) + "\n")

nsga3Succesful = (confidences[nsga3ConfidenceNames] > targetConfidence).all(axis=1)
customSuccesful = (confidences[customConfidenceNames] > targetConfidence).all(axis=1)

print("nsga3 succeful: \n" + str(nsga3Averages[nsga3Succesful].mean()))
print("custom succeful: \n" + str(customAverages[customSuccesful].mean()))

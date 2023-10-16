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

    colors = plt.cm.viridis(np.linspace(0, 1, nColumns))
    colors = np.append(colors[0::2], colors[1::2], axis=0)

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

# read dataframe from csv
results = readFromCsv('../results/resultsAll.csv')

if evaluate:
    evaluateAdaptations(results)

# select sub-dataframes to plot
confidences = results[["nsga3_confidence", "custom_confidence"]]
nReqs = len(results["nsga3_confidence"][0])

# decompose arrays columns into single values columns
if nReqs > 1:
    nsga3Confidences = pd.DataFrame(results['nsga3_confidence'].to_list(),
                                    columns=['nsga3_req_' + str(i) for i in range(nReqs)])
    customConfidences = pd.DataFrame(results['custom_confidence'].to_list(),
                                     columns=['custom_req_' + str(i) for i in range(nReqs)])
    confidences = pd.concat([nsga3Confidences, customConfidences], axis=1)

scores = results[["nsga3_score", "custom_score"]]
times = results[["nsga3_time", "custom_time"]]

personalizedBoxPlot(confidences, "Confidences comparison", 30)
personalizedBoxPlot(scores, "Score comparison")
personalizedBoxPlot(times, "Execution time comparison")

customDataset = pd.read_csv('../results/customDataset.csv')
nsga3Dataset = pd.read_csv('../results/nsga3Dataset.csv')

customAverages = customDataset.loc[:, "req_0":].mean(axis=1)
nsga3Averages = nsga3Dataset.loc[:, "req_0":].mean(axis=1)

print(customAverages.mean())
print(nsga3Averages.mean())

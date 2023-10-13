import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval

def personalizedBoxPlot(data, name, rotation = 0):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = ax.boxplot(data, patch_artist=True,
                    notch='True', vert=True)

    colors = ['#0000FF', '#00FF00',
              '#FFFF00', '#FF00FF',
              '#0000FF', '#00FF00',
              '#FFFF00', '#FF00FF']

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
    ax.set_xticklabels(data.columns, rotation = rotation)

    # Adding title
    plt.title(name)

    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    fig.show()


results = pd.read_csv('../results/results.csv')
columns = ["nsga3_adaptation", "custom_adaptation", "nsga3_confidence", "custom_confidence"]

for c in columns:
    results[c] = results[c].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=' '))

confidences = results[["nsga3_confidence", "custom_confidence"]]
nReqs = len(results["nsga3_confidence"][0])
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
import matplotlib.pyplot as plt
import pandas as pd


def a_dominates_b(a, b, to_min, to_max):
    n_better = 0

    for f in to_min:
        if a[f] > b[f]:
            return False
        n_better += a[f] < b[f]

    for f in to_max:
        if a[f] < b[f]:
            return False
        n_better += a[f] > b[f]

    if n_better > 0:
        return True
    return False

def find_skyline_brute_force(df, to_min = [], to_max = []):
    rows = df.to_dict(orient='index')
    skyline = set()
    for i in rows:
        dominated = False
        for j in skyline:
            if a_dominates_b(rows[j], rows[i], to_min, to_max):
                dominated = True
                break
        if not dominated:
            skyline.add(i)

    return pd.Series(df.index.isin(skyline), index=df.index)


def plot_skyline(df, to_min = [], to_max = []):
    skyline = find_skyline_brute_force(df, to_min, to_max)

    colors = skyline.map({True: 'C1', False: 'C0'})
    ax = df.plot.scatter(x='x', y='y', c=colors, alpha=0.8)
    ax.set_title('skylines')
    plt.show()
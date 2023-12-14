import os
import warnings
import pandas as pd

# suppress all warnings
warnings.filterwarnings("ignore")

files = os.listdir("./")
files.sort()
csvs = list()

for file in files:
    if file == "merge.csv":
        continue
    elif file.endswith('.csv'):
        csvs.append(file)

if len(csvs) == 0:
    print("There are no csvs")
    exit(0)

df = pd.read_csv(os.path.join("./", csvs.pop(0)))

for csv in csvs:
    new_df = pd.read_csv(os.path.join("./", csv))
    frames = [df, new_df]
    df = pd.concat(frames)

df.to_csv("merge.csv", index=False)

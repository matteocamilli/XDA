import pandas as pd
import bambi
import pymc3 as pm

df = pd.read_csv("./0-1.csv")

df = df.drop(columns=["req_1", "req_2", "req_3", "req_4", "req_5"])
df['req_0'] = df['req_0'].astype(float)
df.rename(columns={'req_0': 'y'}, inplace=True)

# copy the data
df_old = df
df = df_old.copy()


# apply normalization techniques
for column in df.columns:
    df[column] = df[column] / df[column].abs().max()

formula = "y ~ power + cruise_speed + bandwidth + quality + illuminance + " \
          "smoke_intensity + obstacle_size + obstacle_distance + firm_obstacle"

"""
def bma(sample):
    reg = bambi.Model(formula, sample).fit()
    return reg
"""


def bma(sample):
    with pm.Model() as model:
        pm.glm.GLM.from_formula(formula, sample, family=pm.glm.families.Binomial())
    with model:
        trace = pm.sample()  # draws=5000, tune=1000, cores=2)

    pm.traceplot(trace)
    pm.summary(trace)

    return trace


test = bma(df)

print("HI")

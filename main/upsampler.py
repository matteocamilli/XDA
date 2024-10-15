import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv('../datasets/uavv2.csv')

reqs = ["req_0", "req_1", "req_2", "req_3", "req_4", "req_5", "req_6", "req_7", "req_8", "req_9", "req_10", "req_11"]

X = df.iloc[:, :7]
y = df.iloc[:, 7:]

over_sampler = RandomOverSampler()
under_sampler = RandomUnderSampler()

for col in y.columns:
    X_resampled_over, y_resampled_over = over_sampler.fit_resample(X, y[col])

    X_resampled, y_resampled = under_sampler.fit_resample(X_resampled_over, y_resampled_over)

    balanced_data = pd.concat([X_resampled, y_resampled], axis=1)

    req_column = balanced_data.iloc[:, 7:].dropna(axis=1, how='all').columns[0]
    filtered_data = balanced_data[
        ['formation', 'flying_speed', 'countermeasure', 'weather', 'day_time', 'threat_range', '#threats', req_column]]

    filtered_data.rename(columns={req_column: col}, inplace=True)

    filtered_data.dropna(inplace=True)

    filtered_data.to_csv(f'bilanciato_{col}.csv', index=False)

    counts = filtered_data[col].value_counts()

    print(f"Dataset {col}:")
    print(counts)
    print()

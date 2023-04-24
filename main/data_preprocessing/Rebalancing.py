import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def get_class_ratio(input_data):
    n_data = len(input_data)
    n_defective = sum(input_data)
    n_clean = n_data - n_defective
    print('From the total of', n_data, 'files, there are:')
    print(n_defective, 'defective files', '(' + str(np.round(n_defective * 1.0 / n_data * 100, 2)) + '%)')
    print(n_clean, 'clean files', '(' + str(np.round(n_clean * 1.0 / n_data * 100, 2)) + '%)')


def overSampling(X, y):
    # Apply the Over-Sampling technique
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_OVER_train, y_OVER_train = oversample.fit_resample(X, y)

    # Find a class ratio of the over-sampled training dataset
    get_class_ratio(y_OVER_train)


def underSampling(X, y):
    # Apply the Under-Sampling technique
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_UNDER_train, y_UNDER_train = undersample.fit_resample(X, y)

    # Find a class ratio of the under-sampled training dataset
    get_class_ratio(y_UNDER_train)


def syntheticOverSampling(X, y):
    # Apply the SMOTE technique
    oversample_SMOTE = SMOTE(sampling_strategy='minority')
    X_SMOTE_train, y_SMOTE_train = oversample_SMOTE.fit_resample(X, y)

    # Find a class ratio of the SMOTE-ed training dataset
    get_class_ratio(y_SMOTE_train)
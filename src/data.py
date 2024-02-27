import os
import datetime as dt
import pandas as pd
import numpy as np

NUM_DAYS_IN_WEEK = 7


def load_data(data_file: str):
    package_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(package_dir, 'data')
    os.chdir(data_dir)
    return pd.read_csv(data_file)


def map_into_numeric(data: pd.DataFrame, mappings: dict):
    for col_name in mappings:
        data = data.replace({col_name: mappings[col_name]})
    return data


def is_data_full(data: pd.DataFrame):
    first_day = dt.datetime.strptime(data.index[0], '%Y-%m-%d')
    last_day = dt.datetime.strptime(data.index[-1], '%Y-%m-%d')
    if first_day + dt.timedelta(len(data) - 1) == last_day:
        return True
    else:
        return False


def sliding_window(elements, y_col, window_size: int):
    if len(elements) <= window_size:
        return [], []

    X = {col: [] for col in elements.columns}
    y = []

    for i in range(len(elements) - window_size):
        if elements[y_col][i + window_size] is not None:
            for col in X:
                X[col].append(pd.Series(elements[col][i:i + window_size]))
            y.append(elements[y_col][i + window_size])
    return X, y


def get_X_y(df, input_size=1):
    X, y = sliding_window(df, 'Sales', window_size=NUM_DAYS_IN_WEEK)

    if input_size > 1:
        longer_inputs = {}
        for company in X:
            longer_inputs[company] = []
            for i in range(len(X[company]) - input_size):
                longer_inputs[company].append(np.array(X[company][i:i + input_size]).flatten())
        X = longer_inputs
        y = y[:len(y) - input_size]

    features = []

    for feature in X:
        features.append(
            pd.DataFrame(np.array(X[feature]), columns=[f'{feature}{i}' for i in range(NUM_DAYS_IN_WEEK * input_size)]))

    X = pd.concat(features, axis=1)
    X.index = df.index[NUM_DAYS_IN_WEEK:]
    y = pd.Series(y, index=df.index[NUM_DAYS_IN_WEEK:])
    return X, y

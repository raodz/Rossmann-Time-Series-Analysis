import os
import datetime as dt
import pandas as pd


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

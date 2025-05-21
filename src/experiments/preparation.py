import os
from src.preprocessing_data import preprocess_data
from src.utils import RESULTS_DIR_NAME
from src.logging import (setup_logging, log_program_start,
                         log_data_preparation_start,
                         log_data_preparation_complete)


def prepare_environment():
    """Prepare execution environment - directories, logs."""
    os.makedirs(RESULTS_DIR_NAME, exist_ok=True)
    setup_logging(RESULTS_DIR_NAME)
    log_program_start()


def prepare_experiment_data(config, excluded_features, exclude_sundays):
    """Prepare data for experiment."""
    log_data_preparation_start()

    X_train, y_train, X_test, y_test = preprocess_data(
        config['data_path'],
        y_col=config['data_params']['y_col'],
        prc_samples_for_test=config['data_params']['prc_samples_for_test'],
        numeric_mappings=config['data_params']['numeric_mappings'],
        excluded_features=excluded_features,
        exclude_sundays_flag=exclude_sundays
    )

    # Exclude Sunday column if needed
    if exclude_sundays and 'DayIsSunday' in X_train.columns:
        X_train = X_train.drop(columns=['DayIsSunday'])
        X_test = X_test.drop(columns=['DayIsSunday'])

    log_data_preparation_complete(X_train.columns.tolist())

    return X_train, y_train, X_test, y_test

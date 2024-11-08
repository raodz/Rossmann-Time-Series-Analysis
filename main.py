import os
import yaml
import logging
from src.data import preprocess_data
from src.constants import RESULTS_DIR_NAME
from models.arima_model import ARIMAModel
from models.linear_regression_model import LinearRegressionModel
from models.dummy_model import DummyModel
from src.setup_logging import setup_logging
from src.plots import plot_test_predictions


def main():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    results_dir_path = os.path.join('..', RESULTS_DIR_NAME)
    os.makedirs(results_dir_path, exist_ok=True)
    setup_logging(results_dir_path)
    logging.info("Program execution started")

    logging.info("Preparing data")
    X_train, y_train, X_test, y_test = preprocess_data(config['data_path'],
                                                       y_col=config['data_params']['y_col'],
                                                       prc_samples_for_test=config['data_params']['prc_samples_for_test'],
                                                       numeric_mappings=config['data_params']['numeric_mappings'])
    logging.info("Data preparation completed")

    models = {
        'ARIMA': ARIMAModel(config),
        'Linear Regression': LinearRegressionModel(config),
        'Dummy Model': DummyModel(config)
    }

    rmse_results = {}
    predictions = {}

    for name, model in models.items():
        logging.info(f"Training {name} model")
        model.fit(X_train, y_train)
        logging.info(f"{name} model trained")

        rmse = model.evaluate(X_test, y_test)
        rmse_results[name] = rmse
        predictions[name] = model.predict(X_test)
        logging.info(f"RMSE for {name} model: {rmse}")

    logging.info("Comparing model performance:")
    for name, rmse in rmse_results.items():
        logging.info(f"{name}: RMSE = {rmse}")

    # Create plot only for ARIMA model
    y_pred = predictions['ARIMA']
    plot_path = os.path.join(
        results_dir_path,
        config['plot_params']['title']
    )
    plot_title = config['plot_params']['title']
    plot_figsize = config['plot_params']['figsize']

    plot_test_predictions(
        y_test,
        y_pred,
        save_path=plot_path,
        title=plot_title,
        figsize=plot_figsize
    )
    logging.info(f"Plot for ARIMA model saved at {plot_path}")

    logging.info("Program execution completed")


if __name__ == '__main__':
    main()

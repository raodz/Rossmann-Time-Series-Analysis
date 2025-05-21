import logging


def log_experiment_parameters(excluded_features, exclude_sundays):
    """Log experiment configuration parameters."""
    if excluded_features:
        logging.info(f"Features to exclude: {excluded_features}")
    if exclude_sundays:
        logging.info("Excluding Sundays from analysis")


def log_experiment_results(experiment_name, rmse_results):
    """Log experiment results."""
    logging.info(f"Results for experiment: {experiment_name}")
    for name, rmse in rmse_results.items():
        logging.info(f"{name}: RMSE = {rmse}")


def log_model_training_start(model_name):
    """Log when model training starts."""
    logging.info(f"Training {model_name} model")


def log_model_training_complete(model_name):
    """Log when model training is complete."""
    logging.info(f"{model_name} model trained")


def log_model_performance(model_name, rmse):
    """Log model evaluation results."""
    logging.info(f"RMSE for {model_name} model: {rmse}")


def log_plot_saved(model_name, experiment_name, plot_path):
    """Log when a plot is saved."""
    logging.info(f"Plot for {model_name} model ({experiment_name}) saved at {plot_path}")


def log_experiment_start(experiment_name):
    """Log when an experiment starts."""
    logging.info(f"=== Starting experiment: {experiment_name} ===")


def log_experiment_complete(experiment_name):
    """Log when an experiment is complete."""
    logging.info(f"=== Experiment completed: {experiment_name} ===")


def log_data_preparation_start():
    """Log when data preparation starts."""
    logging.info("Preparing data")


def log_data_preparation_complete(features):
    """Log when data preparation is complete."""
    logging.info("Data preparation completed")
    logging.info(f"Features used in the model: {features}")


def log_program_start():
    """Log when the program starts."""
    logging.info("Program execution started")


def log_program_complete():
    """Log when the program completes."""
    logging.info("Program execution completed")


def log_experiments_running():
    """Log when experiments are running."""
    logging.info("Running experiments")
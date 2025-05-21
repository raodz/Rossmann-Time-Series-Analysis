from src.logging import (
    log_experiment_start,
    log_experiment_complete,
    log_experiment_parameters,
    log_experiment_results,
    log_experiments_running
)
from src.experiments.preparation import prepare_experiment_data
from src.experiments.models_management import initialize_models, train_and_evaluate_models
from src.visualization.generating_plots import generate_plots


def run_experiment(config, experiment_name, excluded_features=None, exclude_sundays=False):
    """Run an experiment with specified parameters and return results."""
    log_experiment_start(experiment_name)

    # Log experiment parameters
    log_experiment_parameters(excluded_features, exclude_sundays)

    # Prepare data
    X_train, y_train, X_test, y_test = prepare_experiment_data(
        config, excluded_features, exclude_sundays)

    # Train models and evaluate performance
    models = initialize_models(config)
    rmse_results, predictions = train_and_evaluate_models(
        models, X_train, y_train, X_test, y_test)

    # Log results
    log_experiment_results(experiment_name, rmse_results)

    # Generate plots
    generate_plots(config, experiment_name, predictions, y_test, rmse_results)

    log_experiment_complete(experiment_name)

    return rmse_results


def execute_experiments(config, experiment_definitions):
    """Execute a series of experiments based on provided definitions."""
    log_experiments_running()

    all_results = {}
    for exp_name, features, exclude_sundays in experiment_definitions:
        all_results[exp_name] = run_experiment(
            config=config,
            experiment_name=exp_name,
            excluded_features=features,
            exclude_sundays=exclude_sundays
        )

    return all_results
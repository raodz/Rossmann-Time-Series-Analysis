import logging


def generate_summary(all_results):
    """Generate a summary of all experiment results."""
    logging.info("=== Summary of all experiments ===")
    for exp_name, results in all_results.items():
        logging.info(f"Experiment: {exp_name}")
        for model_name, rmse in results.items():
            logging.info(f"  {model_name}: RMSE = {rmse}")


def generate_model_comparison(all_results):
    """Generate a comparison of models across experiments."""
    logging.info("=== Model comparison across experiments ===")
    for exp_name, results in all_results.items():
        if 'ARIMA' in results and 'Linear Regression' in results:
            compare_models_performance(exp_name, results)


def compare_models_performance(experiment_name, results):
    """Compare ARIMA and Linear Regression performance."""
    arima_rmse = results['ARIMA']
    lr_rmse = results['Linear Regression']
    percent_diff = ((lr_rmse - arima_rmse) / arima_rmse) * 100
    logging.info(
        f"Experiment: {experiment_name} - ARIMA is {percent_diff:.2f}% better than Linear Regression")
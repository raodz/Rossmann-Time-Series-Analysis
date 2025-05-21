import os
from src.utils import RESULTS_DIR_NAME
from src.visualization import plot_test_predictions
from src.logging import log_plot_saved


def generate_plots(config, experiment_name, predictions, y_test, rmse_results):
    """Generate plots for all models in an experiment."""
    results_dir_path = os.path.join(RESULTS_DIR_NAME)
    plot_title = config['plot_params']['title']
    plot_figsize = config['plot_params']['figsize']

    for name, y_pred in predictions.items():
        plot_filename = f"{name.replace(' ', '_').lower()}_{experiment_name}.png"
        plot_path = os.path.join(results_dir_path, plot_filename)

        model_plot_title = f"{plot_title} - {name} ({experiment_name})"

        plot_test_predictions(
            y_test,
            y_pred,
            save_path=plot_path,
            title=model_plot_title,
            figsize=plot_figsize
        )

        log_plot_saved(name, experiment_name, plot_path)
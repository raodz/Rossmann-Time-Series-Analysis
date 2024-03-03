import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def visualize_arima_results(test_df, predictions, y_lim):
    """
    Visualize ARIMA model results and evaluate its performance.

    Parameters:
    -----------
    test_df : pandas.DataFrame or pandas.Series
        The actual time series data used for testing the ARIMA model.

    predictions : statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
        The fitted ARIMA model results.

    y_lim: int
        The upper value of the y-axis range in the plot.

    Returns:
    --------
    None

    Notes:
    ------
    This function visualizes the actual time series data alongside the ARIMA model predictions
    with a shaded confidence interval. It also computes and prints the Root Mean Squared Error (RMSE)
    as a measure of the model's accuracy.

    Parameters:
    -----------
    test_df : pandas.DataFrame or pandas.Series
        The actual time series data used for testing the ARIMA model.

    predictions : statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
        The fitted ARIMA model results.

    Examples:
    ---------
    >>> visualize_arima_results(your_test_data, arima_predictions)
    """

    # przedział ufności dla predykcji
    yhat_conf_int = predictions.conf_int(alpha=0.05)

    # Wyświetl wyniki
    # Zwróć uwagę na użycie funkcji fill_between
    plt.plot(test_df, label='Actual')
    plt.plot(pd.Series(predictions.predicted_mean, index=test_df.index), label='ARIMA Predictions')
    plt.ylim(0, y_lim)
    plt.fill_between(test_df.index, yhat_conf_int[:,0], yhat_conf_int[:,1], color="lightgray")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.show()

    # Ocena modelu za pomocą błędu RMSE
    rmse = np.sqrt(mean_squared_error(test_df.values.reshape(-1), predictions.predicted_mean))
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
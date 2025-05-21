from src.models import ARIMAModel, LinearRegressionModel, DummyModel


def initialize_models(config):
    """
    Initialize and return a dictionary of time series prediction models.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        dict: Dictionary mapping model names to initialized model instances
    """
    return {
        'ARIMA': ARIMAModel(config),
        'Linear Regression': LinearRegressionModel(config),
        'Dummy Model': DummyModel(config)
    }
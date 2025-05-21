from src.models import ARIMAModel, LinearRegressionModel, DummyModel
from src.logging import (log_model_training_start,
                         log_model_training_complete,
                         log_model_performance)


def initialize_models(config):
    """Initialize models for training."""
    return {
        'ARIMA': ARIMAModel(config),
        'Linear Regression': LinearRegressionModel(config),
        'Dummy Model': DummyModel(config)
    }


def train_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    """Train models and evaluate their performance."""
    rmse_results = {}
    predictions = {}

    for name, model in models.items():
        log_model_training_start(name)
        model.fit(X_train, y_train)
        log_model_training_complete(name)

        rmse = model.evaluate(X_test, y_test)
        rmse_results[name] = rmse
        predictions[name] = model.predict(X_test)
        log_model_performance(name, rmse)

    return rmse_results, predictions

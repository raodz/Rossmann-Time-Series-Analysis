from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np

from src.models import BaseModel


class ARIMAModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config['arima_params']
        self.autoarima_params = None
        self.y_train = None
        self.with_exog = True

    def fit(self, X_train, y_train):
        # Save y_train for potential fallback
        self.y_train = y_train

        # Convert all data to appropriate numeric types to avoid dtype issues
        X_train_numeric = X_train.astype(float)
        y_train_numeric = pd.Series(y_train).astype(float)

        try:
            # First try to fit auto_arima with exogenous variables
            self.autoarima_params = auto_arima(
                y=y_train_numeric,
                X=X_train_numeric,
                **self.config
            )
            self.with_exog = True

            # Fit ARIMA model with parameters from auto_arima
            self.model = ARIMA(
                y_train_numeric,
                exog=X_train_numeric,
                order=self.autoarima_params.order,
                seasonal_order=self.autoarima_params.seasonal_order
            ).fit()

        except Exception as e:
            print(f"Error fitting ARIMA with exogenous variables: {e}")
            print("Trying to fit ARIMA without exogenous variables...")

            try:
                # Try fitting without exogenous variables
                # Create a copy of config without exog-related parameters
                config_no_exog = self.config.copy()

                # Remove any parameters related to exogenous variables if they exist
                exog_params = ['exog', 'X']
                for param in exog_params:
                    if param in config_no_exog:
                        del config_no_exog[param]

                self.autoarima_params = auto_arima(
                    y=y_train_numeric,
                    **config_no_exog
                )
                self.with_exog = False

                # Fit ARIMA model without exogenous variables
                self.model = ARIMA(
                    y_train_numeric,
                    order=self.autoarima_params.order,
                    seasonal_order=self.autoarima_params.seasonal_order
                ).fit()

            except Exception as e:
                print(f"Error fitting ARIMA without exogenous variables: {e}")
                print("Falling back to simple ARIMA(1,1,1) model")

                try:
                    # Last resort: try a simple ARIMA(1,1,1) model
                    self.with_exog = False
                    self.model = ARIMA(
                        y_train_numeric,
                        order=(1, 1, 1)
                    ).fit()
                    self.autoarima_params = None

                except Exception as e:
                    print(f"Error fitting simple ARIMA model: {e}")
                    print("Using moving average prediction as fallback")
                    self.model = None
                    self.autoarima_params = None

    def predict(self, X_test):
        if self.model is None:
            # Fallback to moving average if all ARIMA attempts failed
            window_size = 7  # one week
            return pd.Series(self.y_train).rolling(
                window=window_size).mean().iloc[-1] * np.ones(len(X_test))

        # Convert X_test to appropriate numeric type
        X_test_numeric = X_test.astype(float) if self.with_exog else None

        # Use the appropriate predict method based on whether we have exogenous variables
        if self.with_exog:
            return self.model.forecast(steps=len(X_test), exog=X_test_numeric)
        else:
            return self.model.forecast(steps=len(X_test))
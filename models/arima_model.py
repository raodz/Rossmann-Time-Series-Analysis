from models.base_model import BaseModel
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config['arima_params']
        self.autoarima_params = None

    def fit(self, X_train, y_train):
        self.autoarima_params = auto_arima(
            y=y_train,
            X=X_train,
            **self.config
        )

        self.model = ARIMA(
            y_train,
            exog=X_train,
            order=self.autoarima_params.order,
            seasonal_order=self.autoarima_params.seasonal_order
        ).fit()

    def predict(self, X_test):
        return self.model.forecast(steps=len(X_test), exog=X_test)

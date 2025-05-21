from sklearn.metrics import mean_squared_error
import numpy as np


class BaseModel:
    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse

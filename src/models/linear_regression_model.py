from sklearn.linear_model import LinearRegression

from src.models import BaseModel


class LinearRegressionModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config['linear_regression_params']
        self.model = LinearRegression(**self.config)

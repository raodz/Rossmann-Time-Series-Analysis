from models.base_model import BaseModel
from sklearn.dummy import DummyRegressor


class DummyModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config['dummy_model_params']
        self.model = DummyRegressor(
            strategy=self.config['strategy'],
            constant=self.config.get('constant_value', None)
        )

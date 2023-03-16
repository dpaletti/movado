from abc import ABC
from river.base.regressor import Regressor


class Model(ABC):
    _model: Regressor

    def __init__(self, **kwargs):
        pass

    def get_model(self):
        return self._model

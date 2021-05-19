from abc import ABC, abstractmethod
import river as rv


class Model(ABC):
    _model: rv.base.regressor

    def __init__(self, **kwargs):
        pass

    def get_model(self):
        return self._model

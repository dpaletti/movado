import abc
from abc import abstractmethod
from typing import List
from pathlib import Path


class Estimator(abc.ABC):
    @abstractmethod
    def __init__(self, debug=False):
        self._debug = debug

    @abstractmethod
    def predict(self, X: List[float]) -> List[float]:
        pass

    @abstractmethod
    def train(self, X: List[float], y: List[float]) -> None:
        pass

    @abstractmethod
    def get_error(self) -> float:
        pass

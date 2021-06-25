import abc
from abc import abstractmethod
from typing import List


class Estimator(abc.ABC):
    @abstractmethod
    def predict(self, X: List[float]) -> List[float]:
        pass

    @abstractmethod
    def train(self, X: List[float], y: List[float]) -> None:
        pass

    @abstractmethod
    def get_error(self) -> float:
        pass

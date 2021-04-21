import abc
from abc import abstractmethod
from typing import Dict, List


class Estimator(abc.ABC):
    @abstractmethod
    def estimate(self, design_point) -> float:
        pass

    @abstractmethod
    def train(self, point: List[float], metrics: List[float]) -> None:
        pass

    @abstractmethod
    def get_accuracy(self):
        pass

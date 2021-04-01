import abc
from abc import abstractmethod
import numpy as np


class Estimator(abc.ABC):
    default: str = ""  # TODO: set default estimator

    @abstractmethod
    def estimate(self, design_point) -> float:
        pass

    @abstractmethod
    def train(self, example) -> None:
        pass

    @abstractmethod
    def get_accuracy(self):
        pass

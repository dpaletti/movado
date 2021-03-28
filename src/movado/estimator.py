import abc
from abc import abstractmethod
import numpy as np


class Estimator(abc.ABC):
    default: str = ""  # TODO: set default estimator

    @abstractmethod
    def estimate(self, design_point) -> float:
        pass

    def get_dataset(self) -> "np.ndarray":
        pass

    def train(self, example) -> None:
        pass

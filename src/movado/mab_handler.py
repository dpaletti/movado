from typing import List
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np


class MabHandler(ABC):
    def __init__(self, debug: bool = False):
        self._mab = None
        self._last_predict_probability: float = -1
        self._debug = debug
        self._sample_prefix = ""  # if changing this add trailing whitespace
        if debug:
            self.__costs: List[float] = []
            Path("movado_debug").mkdir(exist_ok=True)
            self.__mab_debug = "movado_debug/mab.csv"
            Path(self.__mab_debug).open("w").close()
            Path(self.__mab_debug).open("a").write("Mean Reward")

    @abstractmethod
    def predict(self, context: List[float]) -> float:
        pass

    def learn(
        self,
        action: float,
        cost: float,
        context: List[float],
        forced_predict_probability: int = None,
    ) -> None:
        sample: str = (
            self._sample_prefix
            + str(action)
            + ":"
            + str(cost)
            + ":"
            + (
                str(self._last_predict_probability)
                if not forced_predict_probability
                else str(forced_predict_probability)
            )
            + " | "
        )
        for feature in context:
            sample += str(feature) + " "
        self._mab.learn(sample)
        if self._debug:
            self.__costs.append(cost)
            Path(self.__mab_debug).open("a").write(str(np.mean(self.__costs)) + "\n")

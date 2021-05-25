from typing import List
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np


class MabHandler(ABC):
    def __init__(self, debug: bool = False, debug_path: str = "mab"):
        self._mab = None
        self._last_predict_probability: float = -1
        self._debug = debug
        self._sample_prefix = ""  # if changing this add trailing whitespace
        self.__costs: List[float] = []
        if debug:
            Path("movado_debug").mkdir(exist_ok=True)
            self.__mab_debug = "movado_debug/" + debug_path + ".csv"
            Path(self.__mab_debug).open("w").close()
            Path(self.__mab_debug).open("a").write(
                "Mean Cost, Action, Context, Probability\n"
            )

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
            Path(self.__mab_debug).open("a").write(
                str(np.mean(self.__costs))
                + ", "
                + str(action)
                + ", "
                + str(context)
                + ", "
                + str(self._last_predict_probability)
                + "\n"
            )

    def get_mean_cost(self) -> float:
        if self.__costs:
            return float(np.mean(self.__costs))
        else:
            return 0

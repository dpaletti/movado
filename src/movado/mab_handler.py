from typing import List, Union
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np


class MabHandler(ABC):
    def __init__(
        self,
        debug: bool = False,
        debug_path: str = "mab",
        skip_debug_initialization: bool = False,
        controller_params=None,
    ):
        self._mab = None
        self._last_predict_probability: float = -1
        self._last_action: Union[int, float] = -1
        self._debug = debug
        self._sample_prefix = ""
        self.__costs: List[float] = []
        self.__controller_params = controller_params
        self.__mab_debug = "movado_debug/" + debug_path + ".csv"
        if debug and not skip_debug_initialization:
            self.initialize_debug()

    def initialize_debug(self):
        Path("movado_debug").mkdir(exist_ok=True)
        Path(self.__mab_debug).open("w").close()
        if self.__controller_params:
            Path(self.__mab_debug).open("a").write(
                "Model_Parameters, Mean Cost, Action, Context, Probability\n"
            )
        else:
            Path(self.__mab_debug).open("a").write(
                "Mean Cost, Action, Context, Probability\n"
            )

    @abstractmethod
    def predict(self, context: List[float]) -> float:
        pass

    def learn(
        self,
        cost: float,
        context: List[float],
        forced_predict_probability: int = None,
        forced_action: Union[int, float] = None,
    ) -> None:
        sample: str = (
            self._sample_prefix
            + (str(self._last_action) if not forced_action else str(forced_action))
            + ":"
            + str(cost)
            + ":"
            + (
                str(self._last_predict_probability)
                if forced_predict_probability is None
                else str(forced_predict_probability)
            )
            + " | "
        )
        for feature in context:
            sample += str(feature) + " "
        self._mab.learn(sample)
        if self._debug:
            self.__costs.append(cost)
            optional_params = (
                (str(self.__controller_params) + ", ")
                if self.__controller_params
                else ""
            )
            Path(self.__mab_debug).open("a").write(
                str(optional_params).replace(",", "")
                + str(np.mean(self.__costs))
                + ", "
                + (
                    str(self._last_action)
                    if forced_action is None
                    else str(forced_action)
                )
                + ", "
                + str(context).replace(",", "")
                + ", "
                + (
                    str(self._last_predict_probability)
                    if forced_predict_probability is None
                    else str(forced_predict_probability)
                )
                + "\n"
            )

    def get_mean_cost(self) -> float:
        if self.__costs:
            return float(np.mean(self.__costs))
        else:
            return 0

    def get_last_predict_probability(self) -> float:
        return self._last_predict_probability

    def set_last_predict_probability(self, last_predict_probability: float) -> None:
        self._last_predict_probability = last_predict_probability

    def get_last_action(self) -> Union[int, float]:
        return self._last_action

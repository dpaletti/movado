import abc
from typing import Optional, List

from vowpalwabbit import pyvw

from movado.mab_implementations_enum import MabImplementation


class MabHandler:
    def __init__(
        self,
        mab_implementation: MabImplementation,
        mab_actions: Optional[int] = None,
        mab_bandwidth: Optional[int] = None,
    ):
        if not mab_actions:
            mab_actions = 100
        if not mab_bandwidth:
            mab_bandwidth = 5
        if mab_implementation is MabImplementation.CATS:
            self.__mab: pyvw.vw = pyvw.vw(
                "cats_pdf "
                + str(mab_actions)
                + "  --bandwidth "
                + str(mab_bandwidth)
                + " --min_value 0 --max_value 100 --chain_hash --coin --epsilon 0.2 -q :: "
            )

    def predict(self, context: List[float]) -> float:
        context_str: str = "|"
        for feature in context:
            context_str += str(feature) + " "
        context_str.strip()
        return self.__mab.predict(
            context_str
        )  # TODO care here the action needs to be sampled

    def train(
        self, action: str, cost: float, context: List[float], probability: float
    ) -> None:
        sample: str = str(action) + ":" + str(cost) + ":" + str(probability) + "|"
        for feature in context:
            sample += str(feature) + " "
        self.__mab.learn(sample)

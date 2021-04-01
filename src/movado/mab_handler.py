import abc
from typing import Optional

from vowpalwabbit import pyvw

from movado.mab_implementations import MabImplementation


class MabHandler(abc.ABC):
    @abc.abstractmethod
    def __init__(self, mab_implementation: MabImplementation ,mab_actions: Optional[int] = None, mab_bandwidth: Optional[int] = None)
        if not mab_actions:
            mab_actions = 100
        if not mab_bandwidth:
            mab_bandwidth= 5
        if mab_implementation is MabImplementation.CATS:
            self.__mab = pyvw.vw(
                "cats_pdf "
                + str(mab_actions)
                + "  --bandwidth "
                + str(mab_bandwidth)
                + " --min_value 0 --max_value 100 --chain_hash --coin --epsilon 0.2 -q :: "
            )

    def predict(self, context: "np.ndarray") -> float:
        context: str = "|"
        for feature in context:
            context += str(feature) + " "
        context.strip()
        return self.__mab.predict(context) #TODO care here the action needs to be sampled

    def train(self, reward: float, context: "np.ndarray"):
        sample: str = ""
        for fe

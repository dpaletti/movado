from typing import List, Tuple

from vowpalwabbit import pyvw
from movado.mab_handler import MabHandler
import numpy as np


class MabHandlerCB(MabHandler):
    def __init__(self, debug: bool = False, epsilon: float = 0.2):
        super().__init__(debug)
        self._mab = pyvw.vw("--cb_explore 2 " + "--cover " + str(3))

    def predict(self, context: List[float]) -> int:
        context_str: str = "| "
        for feature in context:
            context_str += str(feature) + " "
        context_str.strip()
        probability_distribution: Tuple[float, float] = self._mab.predict(context_str)
        self._last_predict_probability = np.max(probability_distribution)
        return np.argmax(probability_distribution) + 1

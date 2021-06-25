from typing import List, Tuple

from vowpalwabbit import pyvw
from movado.mab_handler import MabHandler
import numpy as np
import random


class MabHandlerCB(MabHandler):
    def __init__(
        self,
        arms: int,
        debug: bool = False,
        cover: float = 3,
        controller_params: dict = None,
        debug_path: str = "mab",
    ):
        super().__init__(
            debug, controller_params=controller_params, debug_path=debug_path
        )
        self._mab = pyvw.vw("--cb_explore " + str(arms) + " --cover " + str(cover))

    def predict(self, context: List[float]) -> int:
        context_str: str = "| "
        for feature in context:
            context_str += str(feature) + " "
        context_str.strip()
        prediction: Tuple[int, float] = self.sample_probability_mass_function(
            self._mab.predict(context_str)
        )
        self._last_predict_probability = prediction[1]
        return prediction[0] + 1

    @staticmethod
    def sample_probability_mass_function(
        probability_mass_function: List[float],
    ) -> Tuple[int, float]:
        total = sum(probability_mass_function)
        scale = 1 / total
        probability_mass_function = [x * scale for x in probability_mass_function]
        draw = random.random()
        sum_prob = 0.0
        for index, prob in enumerate(probability_mass_function):
            sum_prob += prob
            if sum_prob > draw:
                return index, prob

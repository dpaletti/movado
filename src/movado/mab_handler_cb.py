from typing import List, Tuple, Union

from vowpalwabbit import pyvw
from movado.mab_handler import MabHandler
import random


class MabHandlerCB(MabHandler):
    def __init__(
        self,
        arms: int,
        cover: int = 3,
        debug: bool = False,
        controller_params: dict = None,
        debug_path: str = "mab",
        skip_debug_initialization: bool = False,
    ):
        super().__init__(
            debug,
            controller_params=controller_params,
            debug_path=debug_path,
            skip_debug_initialization=skip_debug_initialization,
        )
        self._mab = pyvw.vw(
            "--cb_explore "
            + str(arms)
            + " --cover "
            + str(cover)
            + " --quiet"
            + " --random_seed 0"
        )

    def predict(
        self, context: List[float], probability: bool = False
    ) -> Union[int, Tuple[int, float]]:
        context_str: str = "| "
        for feature in context:
            context_str += str(feature) + " "
        context_str.strip()
        prediction: Tuple[int, float] = self.sample_probability_mass_function(
            self._mab.predict(context_str)
        )
        self._last_predict_probability = prediction[1]
        self._last_action = prediction[0]
        if probability:
            return (
                (prediction[0], self._last_predict_probability)
                if self._last_action == 0
                else (prediction[0], 1 - self._last_predict_probability)
            )
        return prediction[0]

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

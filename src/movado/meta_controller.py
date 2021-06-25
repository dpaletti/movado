from collections import OrderedDict
from typing import Union, Callable, List, Dict, Any, Optional

from movado.controller import Controller
from movado.estimator import Estimator
from movado.mab_handler_cb import MabHandlerCB
from pygmo.core import hypervolume
import numpy as np
import itertools

# This inputs are used to populate the symbol table for class retrieval
# noinspection PyUnresolvedReferences
from movado.mab_controller import MabController  # pylint: disable=unused-import

# noinspection PyUnresolvedReferences
from movado.distance_controller import (
    DistanceController,
)  # pylint: disable=unused-import

# noinspection PyUnresolvedReferences


class MetaController(Controller):
    def __init__(
        self,
        controller,  # class reference
        estimator: Estimator,
        exact_fitness: Callable[[List[float]], List[float]],
        params: "OrderedDict[str, List[Union[int, float, str]]]",
        problem_dimensionality: int = -1,
        solutions=None,
        self_exact: Optional[object] = None,
        debug=False,
    ):
        if problem_dimensionality == -1:
            raise Exception(
                "Please specify problem dimensionality for MetaController instantiation"
            )
        if solutions is None:
            raise Exception("Please pass an AsyncList of solutions")

        super(MetaController, self).__init__(
            estimator=estimator,
            exact_fitness=exact_fitness,
            debug=debug,
            self_exact=self_exact,
        )
        self.__debug = debug
        self.__controllers: List[Controller] = self.__get_controllers(
            params,
            controller,
            estimator,
            exact_fitness,
            self_exact=self_exact,
            debug=debug,
        )
        self.__mab: MabHandlerCB = MabHandlerCB(
            len(self.__controllers),
            debug=debug,
            debug_path="meta_mab",
            cover=20,
        )
        self.__reference_point: List[float] = [
            1 + (1 / (problem_dimensionality - 1))
        ] * problem_dimensionality
        self.__last_hv: Optional[float] = 0
        self.__solutions: AsyncList = solutions
        self.__last_action: int = -1
        self.__last_point: List[float] = []
        if self._debug:
            self.initialize_debug()

    def learn(self):
        hv_obj = hypervolume(
            [
                solution / np.linalg.norm(solution, ord=1)
                for solution in self.__solutions
            ]
        )
        hv = hv_obj.compute(self.__reference_point)
        self.__mab.learn(self.__last_action, -(hv - self.__last_hv), self.__last_point)
        self.__last_hv = hv

    def initialize_debug(self):
        self.__controllers[0].initialize_debug()

    @staticmethod
    def __get_controllers(
        params: "OrderedDict[str, List[Union[int, float, str]]]",
        controller,
        estimator: Estimator,
        exact_fitness: Callable[[List[float]], List[float]],
        self_exact: Optional[object],
        debug: bool,
    ) -> List[Controller]:
        controllers: List[Controller] = []
        controller_class = controller
        if not controller_class:
            raise Exception("Controller '" + str(controller) + "' was not found")
        for current_vals in itertools.product(*list(params.values())):
            current_params = {k: v for k, v in zip(params.keys(), current_vals)}
            controllers.append(
                controller_class(
                    estimator=estimator,
                    exact_fitness=exact_fitness,
                    self_exact=self_exact,
                    debug=debug,
                    skip_debug_initialization=True,
                    **current_params
                )
            )
        return controllers

    def compute_objective(self, point: List[int]) -> List[float]:
        self.__last_action = self.__mab.predict(point)
        self.__last_point = point
        return self.__controllers[self.__last_action - 1].compute_objective(
            self.__last_point
        )

    def write_debug(self, debug_info: Dict[str, Any]):
        pass

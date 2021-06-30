from collections import OrderedDict
from typing import Union, Callable, List, Dict, Any, Optional, Tuple
from pathlib import Path

from movado.controller import Controller
from movado.estimator import Estimator
from movado.mab_handler import MabHandler
import numpy as np
import itertools

# This inputs are used to populate the symbol table for class retrieval
# noinspection PyUnresolvedReferences
from movado.mab_controller import MabController  # pylint: disable=unused-import

# noinspection PyUnresolvedReferences
from movado.distance_controller import (
    DistanceController,
)  # pylint: disable=unused-import


class VotingController(Controller):
    def __init__(
        self,
        controller,  # class reference => assumption of homogeneous controllers with different parameters
        estimator: Estimator,
        exact_fitness: Callable[[List[float]], List[float]],
        params: "OrderedDict[str, List[Union[int, float, str]]]",
        self_exact: Optional[object] = None,
        is_point_in_context: bool = False,
        mab_weight: bool = True,
        debug=False,
    ):

        super(VotingController, self).__init__(
            estimator=estimator,
            exact_fitness=exact_fitness,
            debug=debug,
            self_exact=self_exact,
        )
        self.__is_point_in_context = is_point_in_context
        self.__debug = debug
        self.__controllers: List[Controller] = self.__get_controllers(
            params,
            controller,
            estimator,
            exact_fitness,
            self_exact=self_exact,
            mab_weight=mab_weight,
            debug=debug,
        )
        self.__last_winners: List[Controller] = []
        self.__last_decision: int = -1
        self.initialize_debug()
        self.__params = params

    def initialize_debug(self):
        Path(self._controller_debug).open("a").write(
            "Majority_Model_Parameters, Majority_Size, Threshold, Nth_Nearest_Distance, Point, Exec_Time, Error, Estimation\n"
        )

    def learn(
        self,
        is_exact: bool,
        point: List[float] = None,
        exec_time: float = None,
        mab: Optional[Tuple[MabHandler, Union[int, float]]] = None,
        mab_forced_probability: Optional[float] = None,
        mab_forced_action: Optional[Union[int, float]] = None,
        mab_weight: Optional[MabHandler] = None,
        mab_weight_forced_probability: Optional[float] = None,
        mab_weight_forced_action: Optional[Union[int, float]] = None,
        is_point_in_context: bool = True,
    ):
        for controller in self.__controllers:
            if controller in self.__last_winners:
                controller.learn(
                    is_exact=is_exact,
                    point=point,
                    exec_time=exec_time,
                    mab=(controller.get_mab(), self.__last_decision),
                    mab_forced_probability=None,
                    mab_weight=controller.get_weight_mab(),
                    mab_weight_forced_probability=None,
                    is_point_in_context=is_point_in_context,
                )
            else:
                controller.learn(
                    is_exact=is_exact,
                    point=point,
                    exec_time=exec_time,
                    mab=(controller.get_mab(), self.__last_decision),
                    mab_forced_probability=1,
                    mab_forced_action=float(
                        np.mean(
                            [
                                winner.get_mab().get_last_action()
                                for winner in self.__last_winners
                            ]
                        )
                    ),
                    mab_weight=controller.get_weight_mab(),
                    mab_weight_forced_probability=None,
                    is_point_in_context=is_point_in_context,
                )

    @staticmethod
    def __get_controllers(
        params: "OrderedDict[str, List[Union[int, float, str]]]",
        controller,
        estimator: Estimator,
        exact_fitness: Callable[[List[float]], List[float]],
        self_exact: Optional[object],
        mab_weight: bool,
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
                    mab_weight=mab_weight,
                    **current_params
                )
            )
        mab: MabHandler = controllers[0].get_mab()
        mab_weight: MabHandler = controllers[0].get_weight_mab()
        if mab:
            mab.initialize_debug()
        if mab_weight:
            mab_weight.initialize_debug()
        return controllers

    def compute_objective(
        self, point: List[int], decision_only: bool = False
    ) -> List[float]:
        decisions = []
        for controller in self.__controllers:
            decisions.append(controller.compute_objective(point, decision_only=True))
        decision = np.mean(decisions)
        if decision >= 0.5 or self._estimator.get_error() == 0.0:
            self.__last_decision = 1
            self.__last_winners = [
                controller_decision[0]
                for controller_decision in zip(self.__controllers, decisions)
                if controller_decision[1] == 1
            ]
            out, exec_time = self._compute_exact(
                point, is_point_in_context=self.__is_point_in_context
            )
        else:
            self.__last_decision = 0
            self.__last_winners = [
                controller_decision[0]
                for controller_decision in zip(self.__controllers, decisions)
                if controller_decision[1] == 0
            ]
            out, exec_time = self._compute_estimated(
                point,
                is_point_in_context=self.__is_point_in_context,
            )

        if self._debug:
            params = self.__last_winners[0].get_parameters()
            self.write_debug(
                {
                    params[0]: params[1],
                    "Majority_Size": len(self.__last_winners),
                    "Point": point,
                    "Exec_Time": exec_time,
                    "Error": self._estimator.get_error(),
                    "Estimation": int(not self.__last_decision),
                }
            )
        return out

    def write_debug(self, debug_info: Dict[str, Any]):
        Path(self._controller_debug).open("a").write(
            str(debug_info["Model_Parameters"])
            + ", "
            + str(debug_info["Majority_Size"])
            + ", "
            + str(debug_info["Point"])
            + ", "
            + str(debug_info["Exec_Time"])
            + ", "
            + str(debug_info["Error"])
            + ", "
            + str(debug_info["Estimation"])
            + "\n"
        )

    def get_mab(self):
        raise Exception("'get_mab': Unsupported method in voting controller")

    def get_weight_mab(self):
        raise Exception("'get_weight_mab': Unsupported method in voting controller")

    def get_parameters(self):
        return self.__params

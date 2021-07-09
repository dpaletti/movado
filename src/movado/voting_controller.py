from collections import OrderedDict
from typing import Union, Callable, List, Dict, Any, Optional, Tuple
from pathlib import Path

from movado.controller import Controller, is_call_exact
from movado.estimator import Estimator
from movado.mab_handler import MabHandler
import numpy as np
import itertools
import random
import scipy.special
import asyncio
import functools
import concurrent.futures
import multiprocessing

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
        voters: int,
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
        self.__is_soft: bool = False
        self.__controllers: List[Controller] = self.__get_controllers(
            params,
            controller,
            estimator,
            exact_fitness,
            voters,
            self_exact=self_exact,
            mab_weight=mab_weight,
            debug=debug,
        )
        self.__last_winners: List[Controller] = []
        self.__last_decision: int = -1
        self.initialize_debug()
        self.__params = params
        random.seed(0)

    def initialize_debug(self):
        Path(self._controller_debug).open("a").write(
            "Majority_Model_Parameters, Majority_Size, Point,"
            + "Exec_Time, Error, Exact_Estimated_Calls, Mean Weight, Estimation\n"
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
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(multiprocessing.cpu_count() / 2)
        )
        loop.set_default_executor(executor)
        for controller in self.__controllers:
            if controller in self.__last_winners:
                loop.run_in_executor(
                    None,
                    functools.partial(
                        controller.learn,
                        is_exact=is_exact,
                        point=point,
                        exec_time=exec_time,
                        mab=(controller.get_mab(), self.__last_decision),
                        mab_forced_probability=None,
                        mab_weight=controller.get_weight_mab(),
                        mab_weight_forced_probability=None,
                        is_point_in_context=is_point_in_context,
                    ),
                )
                # controller.learn(
                #    is_exact=is_exact,
                #    point=point,
                #    exec_time=exec_time,
                #    mab=(controller.get_mab(), self.__last_decision),
                #    mab_forced_probability=None,
                #    mab_weight=controller.get_weight_mab(),
                #    mab_weight_forced_probability=None,
                #    is_point_in_context=is_point_in_context,
                # )
            else:
                loop.run_in_executor(
                    None,
                    functools.partial(
                        controller.learn,
                        is_exact=is_exact,
                        point=point,
                        exec_time=exec_time,
                        mab=(controller.get_mab(), self.__last_decision),
                        mab_forced_probability=1,
                        mab_forced_action=(
                            float(
                                np.mean(
                                    [
                                        winner.get_mab().get_last_action()
                                        for winner in self.__last_winners
                                    ]
                                )
                            )
                            if not self.__is_soft
                            else self.__last_decision
                        ),
                        mab_weight=controller.get_weight_mab(),
                        mab_weight_forced_probability=None,
                        mab_weight_forced_action=None,
                        is_point_in_context=is_point_in_context,
                    ),
                )
                # controller.learn(
                #    is_exact=is_exact,
                #    point=point,
                #    exec_time=exec_time,
                #    mab=(controller.get_mab(), self.__last_decision),
                #    mab_forced_probability=1,
                #    mab_forced_action=(
                #        float(
                #            np.mean(
                #                [
                #                    winner.get_mab().get_last_action()
                #                    for winner in self.__last_winners
                #                ]
                #            )
                #        )
                #        if not self.__is_soft
                #        else self.__last_decision
                #    ),
                #    mab_weight=controller.get_weight_mab(),
                #    mab_weight_forced_probability=None,
                #    mab_weight_forced_action=None,
                #    is_point_in_context=is_point_in_context,
                # )
        executor.shutdown(wait=True)

    def __get_controllers(
        self,
        params: "OrderedDict[str, List[Union[int, float, str]]]",
        controller,
        estimator: Estimator,
        exact_fitness: Callable[[List[float]], List[float]],
        voters: int,
        self_exact: Optional[object],
        mab_weight: bool,
        debug: bool,
    ) -> List[Controller]:
        # TODO random sample indices, not controllers to save memory
        controllers: List[Controller] = []
        controller_class = controller
        if not controller_class:
            raise Exception("Controller '" + str(controller) + "' was not found")
        for current_vals in random.sample(
            list(itertools.product(*list(params.values()))),
            voters,
        ):
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
        if type(controllers[0]) is MabController:
            self.__is_soft = True
        mab: MabHandler = controllers[0].get_mab()
        mab_weight: MabHandler = controllers[0].get_weight_mab()
        if mab:
            mab.initialize_debug()
        if mab_weight:
            mab_weight.initialize_debug()
        return list(random.sample(controllers, voters))

    def __compute_weight_vector(self):
        return scipy.special.softmax(
            [-contr.get_mab().get_mean_cost() for contr in self.__controllers]
        )

    def compute_objective(
        self, point: List[int], decision_only: bool = False
    ) -> List[float]:
        decisions = []
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(multiprocessing.cpu_count() / 2)
        )
        loop.set_default_executor(executor)

        if self.__is_soft:
            for controller in self.__controllers:
                loop.run_in_executor(
                    None,
                    functools.partial(
                        decisions.append,
                        controller.compute_objective(point, probability=True),
                    ),
                )
                # decisions.append(controller.compute_objective(point, probability=True))
            executor.shutdown(wait=True)
            decision = 1 - np.average([d[1] for d in decisions])
        else:
            for controller in self.__controllers:
                loop.run_in_executor(
                    None,
                    functools.partial(
                        decisions.append,
                        controller.compute_objective(point, decision_only=True),
                    ),
                )
                # decisions.append(
                #    controller.compute_objective(point, decision_only=True)
                # )
            executor.shutdown(wait=True)
            decision = np.average(decisions, weights=self.__compute_weight_vector())
        if decision >= 0.5 or self._estimator.get_error() == 0.0:
            self.__last_decision = 1
            if self.__is_soft:
                self.__last_winners = [
                    controller_decision[0]
                    for controller_decision in zip(self.__controllers, decisions)
                    if controller_decision[1][0] == 1
                ]
            else:
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
            if self.__is_soft:
                self.__last_winners = [
                    controller_decision[0]
                    for controller_decision in zip(self.__controllers, decisions)
                    if controller_decision[1][0] == 0
                ]
            else:
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
            params = (
                self.__last_winners[0].get_parameters()
                if self.__last_winners
                else ["Model_Parameters", "Empty Hard Majority"]
            )
            self.write_debug(
                {
                    params[0]: params[1],
                    "Majority_Size": len(self.__last_winners),
                    "Point": point,
                    "Exec_Time": exec_time,
                    "Error": self._estimator.get_error(),
                    "Exact_Estimated_Calls": [
                        is_call_exact.count(True),
                        is_call_exact.count(False),
                    ],
                    "Mean Weight": np.mean(
                        [
                            ctrl.get_weight_mab().get_last_action()
                            for ctrl in self.__controllers
                        ]
                    ),
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
            + str(debug_info["Exact_Estimated_Calls"])
            + ", "
            + str(debug_info["Mean Weight"])
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

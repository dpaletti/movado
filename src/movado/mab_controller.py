from pathlib import Path
from typing import List, Callable, Dict, Any, Optional, Union, Tuple

from movado.controller import Controller
from movado.estimator import Estimator
from movado.mab_handler_cb import MabHandlerCB
from movado.mab_handler_cats import MabHandlerCATS
from movado.controller import is_call_exact


class MabController(Controller):
    def __init__(
        self,
        exact_fitness: Callable[[List[float]], List[float]],
        estimator: Estimator,
        self_exact: Optional[object] = None,
        debug: bool = False,
        skip_debug_initialization=False,
        cover: int = 3,
        mab_weight: bool = True,
        mab_weight_epsilon: float = 0.2,
        mab_weight_bandwidth: int = 1,
        stochastic: bool = False,
    ):
        super().__init__(
            exact_fitness=exact_fitness,
            estimator=estimator,
            self_exact=self_exact,
            debug=debug,
        )
        self.__params = (
            "Model_Parameters",
            {
                "cover": cover,
                "mab_weight_epsilon": mab_weight_epsilon,
                "mab_weight_bandwidth": mab_weight_bandwidth,
            },
        )

        self.__cover = cover
        self.__mab = MabHandlerCB(
            arms=2,
            debug=debug,
            cover=cover,
            controller_params={self.__params[0]: self.__params[1]},
        )
        self.__weight_mab = None
        if mab_weight:
            self.__weight_mab = MabHandlerCATS(
                debug=debug,
                epsilon=mab_weight_epsilon,
                bandwidth=mab_weight_bandwidth,
                controller_params={self.__params[0]: self.__params[1]},
                debug_path="mab_weight",
            )
        self.__is_first_call = True

        self.__exact_calls_cache = {}
        self.__stochastic = stochastic
        if self._debug and not skip_debug_initialization:
            self.initialize_debug()

    def initialize_debug(self):
        Path(self._controller_debug).open("a").write(
            "Point, Exec_Time, Error, Exact_Estimated_Calls\n"
        )

    def compute_objective(
        self,
        point: List[int],
        decision_only: bool = False,
        probability=False,
    ) -> Union[List[float], int, Tuple[int, float]]:
        cached_value = self.__exact_calls_cache.get(tuple(point))
        if not self.__stochastic and cached_value:
            print("Using Cached Exact Value...")
            if self._debug:
                self.write_debug(
                    {
                        "Point": point,
                        "Exec_Time": 0,
                        "Error": self._estimator.get_error(),
                        "Estimation": 0,
                        "Exact_Estimated_Calls": [
                            is_call_exact.count(True),
                            is_call_exact.count(False),
                        ],
                    }
                )
            return cached_value
        decision = self.__mab.predict(
            self._compute_controller_context(point), probability=probability
        )
        accuracy = self._estimator.get_error()
        if probability:
            return (decision[0], decision[1]) if accuracy != 0.0 else (1, decision[1])
        if decision_only:
            return decision
        if decision == 1 or accuracy == 0.0:
            out, exec_time = self._compute_exact(
                point,
                (self.__mab, 1),
                1 if accuracy == 0.0 else None,
                self.__weight_mab,
                1 if accuracy == 0.0 else None,
            )
            if not self.__stochastic:
                self.__exact_calls_cache[tuple(point)] = out

        else:
            out, exec_time = self._compute_estimated(
                point, (self.__mab, 0), self.__weight_mab
            )

        # TODO probably this check can be done only once
        if self._debug:
            self.write_debug(
                {
                    "Point": str(point),
                    "Exec_Time": exec_time,
                    "Error": self._estimator.get_error(),
                    "Estimation": 0 if decision == 1 or accuracy == 0.0 else 1,
                    "Exact_Estimated_Calls": str(
                        [
                            is_call_exact.count(True),
                            is_call_exact.count(False),
                        ]
                    ).replace(",", ""),
                }
            )
        self.__is_first_call = False
        return out

    def write_debug(self, debug_info: Dict[str, Any]):
        Path(self._controller_debug).open("a").write(
            str(debug_info["Point"])
            + ", "
            + str(debug_info["Exec_Time"])
            + ", "
            + str(debug_info["Error"])
            + ", "
            + str(debug_info["Exact_Estimated_Calls"])
            + "\n"
        )

    def get_mean_cost(self):
        return self.__mab.get_mean_cost()

    def get_mab(self) -> MabHandlerCB:
        return self.__mab

    def get_weight_mab(self) -> MabHandlerCATS:
        return self.__weight_mab

    def get_parameters(self):
        return self.__params

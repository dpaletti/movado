from pathlib import Path
from typing import List, Callable, Dict, Any

from movado.controller import Controller
from movado.estimator import Estimator
from movado.mab_handler_cb import MabHandlerCB


class MabController(Controller):
    def __init__(
        self,
        exact_fitness: Callable[[List[float]], float],
        estimator: Estimator,
        debug: bool = False,
        epsilon: float = 0.2,
    ):
        super().__init__(exact_fitness, estimator, debug)
        self.__mab = MabHandlerCB(debug, epsilon=epsilon)
        self.__is_first_call = True

        if self._debug:
            Path(self._controller_debug).open("a").write(
                "Point, Exec_Time, Error, Estimation\n"
            )

    def compute_objective(self, point: List[int]) -> float:
        decision = self.__mab.predict([*point, self._estimator.get_error()])
        accuracy = self._estimator.get_error()
        if decision == 2 or accuracy == 0.0:
            out, exec_time = self._compute_exact(
                point,
                (self.__mab, 2) if not self.__is_first_call else None,
                1 if accuracy == 0.0 else None,
            )
        else:
            out, exec_time = self._compute_estimated(point, (self.__mab, 1))

        # TODO probably this check can be done only once
        if self._debug:
            self._write_debug(
                {
                    "Point": point,
                    "Exec_Time": exec_time,
                    "Error": self._estimator.get_error(),
                    "Estimation": 0 if decision == 2 or accuracy == 0.0 else 1,
                }
            )
        self.__is_first_call = False
        return out

    def _write_debug(self, debug_info: Dict[str, Any]):
        Path(self._controller_debug).open("a").write(
            str(debug_info["Point"])
            + ", "
            + str(debug_info["Exec_Time"])
            + ", "
            + str(debug_info["Error"])
            + ", "
            + str(debug_info["Estimation"])
            + "\n"
        )

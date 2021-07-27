from pathlib import Path
from typing import Dict, Any, List, Callable, Optional, Union

from movado.controller import Controller
from movado.estimator import Estimator
from movado.mab_handler import MabHandler


class ShadowController(Controller):
    def __init__(
        self,
        exact_fitness: Callable[[List[float]], List[float]] = None,
        estimator: Estimator = None,
        self_exact: Optional[object] = None,
        debug: bool = False,
        **kwarg
    ):
        super(ShadowController, self).__init__(
            exact_fitness=exact_fitness,
            estimator=estimator,
            self_exact=self_exact,
            debug=debug,
        )
        if debug:
            self.initialize_debug()

    def compute_objective(
        self, point: List[int], decision_only: bool = False
    ) -> Union[List[float], int]:
        if decision_only:
            out = 1
        else:
            out, exec_time = self._compute_exact(point)
        if self._debug:
            self.write_debug(
                {
                    "Point": point,
                    "Exec_Time": exec_time
                    if not decision_only
                    else "No Exec Time available",
                    "Estimation": 0,
                }
            )
        return out

    def initialize_debug(self):
        Path(self._controller_debug).open("a").write("Point, Exec_Time, Estimation\n")

    def write_debug(self, debug_info: Dict[str, Any]):
        Path(self._controller_debug).open("a").write(
            str(debug_info["Point"])
            + ", "
            + str(debug_info["Exec_Time"])
            + ", "
            + str(debug_info["Estimation"])
            + "\n"
        )

    def get_mab(self) -> MabHandler:
        raise Exception("No Mab Available in Shadow Controller")

    def get_weight_mab(self) -> MabHandler:
        raise Exception("No Weight Mab Available in Shadow Controller")

    def get_parameters(self):
        raise Exception("No Parameters Available for Shadow Controler")

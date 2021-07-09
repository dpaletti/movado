from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

from movado.controller import Controller
from movado.estimator import Estimator
from movado.mab_handler import MabHandler


class MultiStepController(Controller):
    def __init__(
        self,
        controller_class,  # class reference => assumption of homogeneous controllers with different parameters
        estimator: Estimator,
        steps: int,
        exact_fitness: Callable[[List[float]], List[float]],
        self_exact: Optional[object] = None,
        debug=False,
        **kwargs
    ):
        super(MultiStepController, self).__init__(
            estimator=estimator,
            exact_fitness=exact_fitness,
            debug=debug,
            self_exact=self_exact,
        )
        controller: Controller = controller_class(
            estimator=estimator,
            exact_fitness=exact_fitness,
            self_exact=self_exact,
            debug=debug,
            **kwargs
        )
        if steps is None or steps < 1:
            raise Exception(
                "Invalid step value: " + str(steps) + " please input a step value >= 1"
            )
        self.__steps = steps
        self.initialize_debug()

    def compute_objective(
        self, point: List[int], decision_only: bool = False
    ) -> List[float]:
        pass

    def initialize_debug(self):
        Path(self._controller_debug).open("a").write(
            "Majority_Model_Parameters, Majority_Size, Point,"
            + "Exec_Time, Error, Exact_Estimated_Calls, Mean Weight, Estimation\n"
        )

    def write_debug(self, debug_info: Dict[str, Any]):
        pass

    def get_mab(self) -> MabHandler:
        pass

    def get_weight_mab(self) -> MabHandler:
        pass

    def get_parameters(self):
        pass

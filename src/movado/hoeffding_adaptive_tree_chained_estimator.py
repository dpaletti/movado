from typing import Optional, Dict, List
from movado.hoeffding_adaptive_tree_estimator import HoeffdingAdaptiveTreeEstimator
import river as rv

from movado.scalers_enum import Scalers
from movado.splitters_enum import Splitters


class HoeffdingAdaptiveTreeChainedEstimator(HoeffdingAdaptiveTreeEstimator):
    def __init__(
        self,
        scaler: Scalers = Scalers.STANDARD,
        rbf_gamma: float = 1.0,
        box_cox: bool = True,
        box_cox_power: float = 1.0,
        grace_period: int = 200,
        splitter: Optional[Splitters] = None,
    ):
        super(HoeffdingAdaptiveTreeChainedEstimator, self).__init__(
            scaler, rbf_gamma, box_cox, box_cox_power, grace_period, splitter
        )
        self._model = rv.multioutput.RegressorChain(self._model)

    def train(self, features: Dict[str, float], target: List[float]) -> None:
        self._model.learn_one()

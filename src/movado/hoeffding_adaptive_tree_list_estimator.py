from typing import Optional

from movado.ensemble_accuracies_enum import EnsembleAccuracies
from movado.hoeffding_adaptive_tree_estimator import HoeffdingAdaptiveTreeEstimator
import river as rv

from movado.scalers_enum import Scalers
from movado.splitters_enum import Splitters


class HoeffdingAdaptiveTreeListEstimator(HoeffdingAdaptiveTreeEstimator):
    def __init__(
        self,
        n_targets: int,
        rbf_gamma: float = 1.0,
        scaler: Scalers = Scalers.STANDARD,
        box_cox: bool = True,
        box_cox_power: float = 1.0,
        grace_period: int = 200,
        splitter: Optional[Splitters] = None,
        ensemble_accuracy: EnsembleAccuracies = EnsembleAccuracies.MIN,
    ):
        super(HoeffdingAdaptiveTreeListEstimator, self).__init__(
            scaler, rbf_gamma, box_cox, box_cox_power, grace_period, splitter
        )
        self._model = [self._model] * n_targets

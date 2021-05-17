from typing import List, Dict

from estimator import Estimator
import river as rv
from abc import ABC, abstractmethod


class HoeffdingAdaptiveTreeEstimator(Estimator, ABC):
    @abstractmethod
    def __init__(
        self,
        eta=2,
        budget=2000,
        debug=False,
    ):
        super(HoeffdingAdaptiveTreeEstimator, self).__init__(debug)
        self._model = rv.preprocessing.StandardScaler()
        self._model |= rv.feature_extraction.RBFSampler()

        self._model |= rv.tree.HoeffdingAdaptiveTreeRegressor(
            leaf_prediction="adaptive"
        )
        models = rv.utils.expand_param_grid(
            self._model,
            {
                "HoeffdingAdaptiveTreeRegressor": {
                    "model_selector_decay": [
                        0.35,
                        0.55,
                        0.75,
                        0.95,
                    ],
                },
                "RBFSampler": {"gamma": [1e-3, 1e-1, 1, 10]},
            },
        )
        self._model = rv.expert.SuccessiveHalvingRegressor(
            models=models,
            metric=rv.metrics.MAE(),
            budget=budget,
            eta=eta,
            verbose=True,
        )

    @staticmethod
    def _X_to_river(X: List[float]) -> Dict[str, float]:
        return dict(zip(["feature_" + str(i) for i in range(len(X))], X))

    @staticmethod
    def _y_to_river(y: List[float]) -> Dict[int, float]:
        return dict(zip(range(len(y)), y))

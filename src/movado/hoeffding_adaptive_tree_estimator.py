import abc
from typing import List, Optional, Union

from movado.estimator import Estimator
import river as rv
from abc import ABC, abstractmethod

from movado.scalers_enum import Scalers
from movado.splitters_enum import Splitters


class HoeffdingAdaptiveTreeEstimator(Estimator, ABC):
    @abstractmethod
    def __init__(
        self,
        scaler: Scalers = Scalers.STANDARD,
        rbf_gamma: float = 1.0,
        box_cox: bool = True,
        feature_space_dimensionality: int = 100,
    ):
        self._model = (
            rv.preprocessing.StandardScaler()
            if scaler == "standard"
            else (
                rv.preprocessing.RobustScaler()
                if scaler == "robust"
                else rv.preprocessing.AdaptiveStandardScaler()
            )
        )
        self._model |= rv.feature_extraction.RBFSampler(
            gamma=rbf_gamma, n_components=feature_space_dimensionality
        )
        if box_cox:
            self._model |= rv.meta.BoxCoxRegressor(
                regressor=rv.tree.HoeffdingAdaptiveTreeRegressor(
                    leaf_prediction="adaptive"
                ),
            )
            models = rv.utils.expand_param_grid(
                self._model,
                {
                    "BoxCoxRegressor": {
                        "regressor": [
                            (
                                rv.tree.HoeffdingAdaptiveTreeRegressor,
                                {
                                    "model_selector_decay": [
                                        0.35,
                                        0.45,
                                        0.55,
                                        0.65,
                                        0.75,
                                        0.85,
                                        0.95,
                                    ],
                                },
                            )
                        ],
                        "power": [0, 0.2, 0.4, 0.8, 1],
                    }
                },
            )
        else:
            # TODO the same as above but on the Tree only without boxcox
            self._model |= rv.tree.HoeffdingAdaptiveTreeRegressor(
                leaf_prediction="adaptive"
            )
        self._model |= rv.expert.SuccessiveHalvingRegressor

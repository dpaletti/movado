from river.preprocessing import StandardScaler
from river.feature_extraction import RBFSampler
from river.tree.hoeffding_adaptive_tree_regressor import HoeffdingAdaptiveTreeRegressor
from river.model_selection import SuccessiveHalvingRegressor
from river.utils import expand_param_grid
from river.metrics import RMSE

from movado.model import Model


class HoeffdingAdaptiveTreeModel(Model):
    def __init__(
        self,
        eta=2,
        budget=2000,
    ):
        super(HoeffdingAdaptiveTreeModel, self).__init__()
        self._model = StandardScaler()
        self._model |= RBFSampler(seed=0)

        self._model |= HoeffdingAdaptiveTreeRegressor(
            leaf_prediction="adaptive", seed=0
        )
        models = expand_param_grid(
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
        self._model = SuccessiveHalvingRegressor(
            models=models,
            metric=RMSE(),
            budget=budget,
            eta=eta,
            verbose=True,
        )

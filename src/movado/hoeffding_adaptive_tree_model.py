import river as rv

from movado.model import Model


class HoeffdingAdaptiveTreeModel(Model):
    def __init__(
        self,
        eta=2,
        budget=2000,
    ):
        super(HoeffdingAdaptiveTreeModel, self).__init__()
        self._model = rv.preprocessing.StandardScaler()
        self._model |= rv.feature_extraction.RBFSampler(seed=0)

        self._model |= rv.tree.HoeffdingAdaptiveTreeRegressor(
            leaf_prediction="adaptive", seed=0
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
            metric=rv.metrics.RMSE(),
            budget=budget,
            eta=eta,
            verbose=True,
        )

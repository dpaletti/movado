import river as rv
import numpy as np

from movado.model import Model


class KernelRegressionModel(Model):
    def __init__(
        self,
        eta=2,
        budget=2000,
    ):
        super(KernelRegressionModel, self).__init__()
        self._model = rv.preprocessing.StandardScaler()
        self._model |= rv.feature_extraction.RBFSampler()

        self._model |= rv.linear_model.LinearRegression()

        models = rv.utils.expand_param_grid(
            self._model,
            {
                "LinearRegression": {
                    "optimizer": [
                        (
                            rv.optim.Adam,
                            {
                                "beta_1": [0.1, 0.01, 0.001],
                                "lr": [0.1, 0.01, 0.001],
                            },
                        )
                    ],
                    "l2": np.linspace(0.001, 0.4, endpoint=True, num=20),
                    "intercept_lr": [
                        rv.optim.schedulers.Optimal(loss=rv.optim.losses.Squared())
                    ],
                    "initializer": [rv.optim.initializers.Normal()],
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

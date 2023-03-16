import river as rv
from river.preprocessing import StandardScaler
from river.feature_extraction import RBFSampler
from river.linear_model import LinearRegression
from river.utils import expand_param_grid

from movado.model import Model


class KernelRegressionModel(Model):
    def __init__(
        self,
        eta=2,
        budget=2000,
    ):
        super(KernelRegressionModel, self).__init__()
        self._model = StandardScaler()
        self._model |= RBFSampler()

        self._model |= LinearRegression()

        models = expand_param_grid(
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
                    "l2": [
                        0.001,
                        0.002,
                        0.004,
                        0.008,
                        0.016,
                        0.032,
                        0.064,
                        0.128,
                        0.256,
                        0.512,
                        0.1024,
                        0.2048,
                        0.4096,
                    ],
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

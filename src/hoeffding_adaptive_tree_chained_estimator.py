from typing import List, Dict
from hoeffding_adaptive_tree_estimator import (
    HoeffdingAdaptiveTreeEstimator,
)
import river as rv

from numbers import Number


class HoeffdingAdaptiveTreeChainedEstimator(HoeffdingAdaptiveTreeEstimator):
    def __init__(
        self,
        eta=2,
        budget=2000,
    ):
        super(HoeffdingAdaptiveTreeChainedEstimator, self).__init__(eta, budget)
        self._model: rv.multioutput.RegressorChain = rv.multioutput.RegressorChain(
            self._model
        )
        self._metric = rv.metrics.RegressionMultiOutput(rv.metrics.MAE())
        rv.base.Regressor

    def train(self, X: List[float], y: List[float]) -> None:
        y_true = self._y_to_river(y)
        y_pred = self._y_to_river(self.predict(X))
        if y_pred:
            self._metric.update(y_true, y_pred)
        self._model.learn_one(
            self._X_to_river(X),
            y_true,  # there may be a bug in river typing
        )

    def predict(self, X: List[float]) -> List[float]:
        if not self._model.order:
            self._model.order = list(range(len(X)))
        return self._model.predict_one(
            self._X_to_river(X),
        ).values()  # Here a dict is returned, wrong typing in river 0.7.0

    def get_error(self) -> float:
        return self._metric.get()

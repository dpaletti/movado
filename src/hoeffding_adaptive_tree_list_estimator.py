from typing import List

from mab_handler_cb.hoeffding_adaptive_tree_estimator import (
    HoeffdingAdaptiveTreeEstimator,
)
import river as rv
import numpy as np


class HoeffdingAdaptiveTreeListEstimator(HoeffdingAdaptiveTreeEstimator):
    def __init__(self, n_targets, eta=2, budget=2000):
        super(HoeffdingAdaptiveTreeListEstimator, self).__init__(eta, budget)
        self._model: [rv.expert.SuccessiveHalvingRegressor] = [self._model] * n_targets
        self._metric = [rv.metrics.MAE()] * n_targets

    def train(self, X: List[float], y: List[float]) -> None:
        for model_i, y_i, metric_i in zip(self._model, y, self._metric):
            metric_i.update(self._y_to_river(y), self._X_to_river(self.predict(X)))
            model_i.learn_one(
                self._X_to_river(X),
                self._y_to_river(y),  # there may be a bug in river typing
            )

    def predict(self, X: List[float]) -> List[float]:
        prediction: List[float] = []
        for model_i in self._model:
            prediction.append(model_i.predict_one(self._X_to_river(X)))
        return prediction

    def get_error(self) -> float:
        return np.mean([i.get() for i in self._metric])  # returns a float not a ndarray

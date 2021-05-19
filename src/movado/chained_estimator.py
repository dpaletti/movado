import river as rv
from typing import List, Dict
from movado.estimator import Estimator
from movado.model import Model


class ChainedEstimator(Estimator):
    def __init__(self, model_to_chain: Model):
        super(ChainedEstimator, self).__init__()
        self._chained_model: rv.multioutput.RegressorChain = (
            rv.multioutput.RegressorChain(model_to_chain.get_model())
        )
        self._metric = rv.metrics.RegressionMultiOutput(rv.metrics.SMAPE())

    def train(self, X: List[float], y: List[float]) -> None:
        y_true = self._y_to_river(y)
        y_pred = self._y_to_river(self.predict(X))
        if y_pred:
            self._metric.update(y_true, y_pred)
        self._chained_model.learn_one(
            self._X_to_river(X),
            y_true,  # there may be a bug in river typing
        )

    def predict(self, X: List[float]) -> List[float]:
        if not self._chained_model.order:
            self._chained_model.order = list(range(len(X)))
        return self._chained_model.predict_one(
            self._X_to_river(X),
        ).values()  # Here a dict is returned, wrong typing in river 0.7.0

    def get_error(self) -> float:
        return self._metric.get() / 200

    @staticmethod
    def _X_to_river(X: List[float]) -> Dict[str, float]:
        return dict(zip(["feature_" + str(i) for i in range(len(X))], X))

    @staticmethod
    def _y_to_river(y: List[float]) -> Dict[int, float]:
        return dict(zip(range(len(y)), y))

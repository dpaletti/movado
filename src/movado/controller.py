from abc import abstractmethod, ABC
from numbers import Number
from typing import Tuple, Optional, Callable, Any, List, Dict, Union
from scipy.stats import PearsonRConstantInputWarning
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy as sp
import time
from pathlib import Path
import river as rv
from collections import deque

from movado.estimator import Estimator
from movado.mab_handler import MabHandler
from movado.chained_estimator import ChainedEstimator

is_call_exact: List[bool] = []  # i-th element True if i-th call is exact


class Controller(ABC):
    @abstractmethod
    def __init__(
        self,
        exact_fitness: Callable[[List[float]], List[float]] = None,
        estimator: Estimator = None,
        self_exact: Optional[object] = None,
        debug: bool = False,
        **kwargs
    ):
        self.__scaler = StandardScaler()
        self.__time_dataset: "deque" = deque()
        self.__error_dataset: "deque" = deque()
        self.__cost_dataset: "deque" = deque()
        self.__to_reshape = False
        self._exact: Callable[[List[float]], List[float]] = exact_fitness
        self._estimator: Estimator = estimator
        self._debug = debug
        self._self_exact = self_exact
        model = rv.preprocessing.StandardScaler()
        model |= rv.feature_extraction.RBFSampler(seed=0)
        model |= rv.linear_model.LinearRegression()
        models = rv.utils.expand_param_grid(
            model,
            {
                "LinearRegression": {
                    "l2": [1e-3, 1e-1, 1, 10],
                },
                "RBFSampler": {"gamma": [1e-3, 1e-1, 1, 10]},
            },
        )
        self.__time_model = rv.expert.SuccessiveHalvingRegressor(
            models=models,
            metric=rv.metrics.RMSE(),
            budget=1,
            eta=10000,
            verbose=True,
        )
        self.__error_model = rv.expert.SuccessiveHalvingRegressor(
            models=models,
            metric=rv.metrics.RMSE(),
            budget=1,
            eta=10000,
            verbose=True,
        )
        self.__fitted_time: "deque" = deque()
        self.__fitted_error: "deque" = deque()
        self.__fitted_time_mean: float = 0
        self.__fitted_error_mean: float = 0
        self.__cost_mean: float = 0
        self.__time_importance: float = 0
        self.__error_importance: float = 0
        if self._debug:
            Path("movado_debug").mkdir(exist_ok=True)
            self._controller_debug = "movado_debug/controller.csv"
            self.__controller_learn_time_debug = (
                "movado_debug/controller_learn_time.csv"
            )
            Path(self.__controller_learn_time_debug).open("w").close()
            Path(self.__controller_learn_time_debug).open("a").write(
                "Learn Time, Estimation\n"
            )
            Path(self._controller_debug).open("w").close()

    @abstractmethod
    def compute_objective(
        self, point: List[int], decision_only: bool = False
    ) -> List[float]:
        pass

    def learn(
        self,
        is_exact: bool,
        point: List[float],
        exec_time: float = None,
        mab: Optional[Tuple[MabHandler, Union[int, float]]] = None,
        mab_forced_probability: Optional[float] = None,
        mab_forced_action: Optional[Union[int, float]] = None,
        mab_weight: Optional[MabHandler] = None,
        mab_weight_forced_probability: Optional[float] = None,
        mab_weight_forced_action: Optional[Union[int, float]] = None,
        is_point_in_context: bool = True,
    ):

        if mab:
            if mab_weight:
                time_weight = (
                    mab_weight.predict(
                        self._compute_weighting_context(self.get_mab().get_mean_cost())
                    )
                    / 100
                )
                mab[0].learn(
                    cost=self.get_time_error_z_score(
                        time_weight=time_weight, error_weight=1 - time_weight
                    ),
                    context=self._compute_controller_context(point),
                    forced_predict_probability=mab_forced_probability,
                    forced_action=mab_forced_action,
                ),
                mab_weight.learn(
                    cost=self.compute_first_order_sensitivity_index(),
                    context=self._compute_weighting_context(
                        self.get_mab().get_mean_cost()
                    ),
                    forced_predict_probability=mab_weight_forced_probability,
                    forced_action=mab_weight_forced_action,
                )
            else:
                mab[0].learn(
                    cost=self.get_time_error_z_score(),
                    context=self._compute_controller_context(point),
                    forced_predict_probability=mab_forced_probability,
                    forced_action=mab_forced_action,
                )

    def _compute_exact(
        self,
        point: List[int],
        mab: Optional[Tuple[MabHandler, Union[int, float]]] = None,
        mab_forced_probability: Optional[int] = None,
        mab_weight: Optional[MabHandler] = None,
        mab_weight_forced_probability: Optional[int] = None,
        is_point_in_context: bool = True,
    ) -> Tuple[List[float], float]:
        global is_call_exact
        is_call_exact.append(True)
        print("(#" + str(len(is_call_exact)) + ") Waiting for Exact Call...")

        if self._self_exact:
            exact, exec_time = Controller.measure_execution_time(
                self._exact, self._self_exact, point
            )
        else:
            exact, exec_time = Controller.measure_execution_time(self._exact, point)

        self.__time_dataset.append(exec_time)
        self.__error_dataset.append(0)
        _, learn_time = Controller.measure_execution_time(
            self.learn,
            is_exact=True,
            point=point,
            exec_time=exec_time,
            mab=mab,
            mab_forced_probability=mab_forced_probability,
            mab_weight=mab_weight,
            mab_weight_forced_probability=mab_weight_forced_probability,
            is_point_in_context=is_point_in_context,
        )
        Path(self.__controller_learn_time_debug).open("a").write(
            str(learn_time) + "," + str(0) + "\n"
        )
        # self.learn(
        #    is_exact=True,
        #    point=point,
        #    exec_time=exec_time,
        #    mab=mab,
        #    mab_forced_probability=mab_forced_probability,
        #    mab_weight=mab_weight,
        #    mab_weight_forced_probability=mab_weight_forced_probability,
        #    is_point_in_context=is_point_in_context,
        # )
        if np.isscalar(exact):
            exact = [exact]
        self._estimator.train(point, exact)
        return exact, exec_time

    @staticmethod
    def __extract_time_context_features(series: List[float]):
        return (
            (np.mean(series), np.var(series), max(series), min(series), series[-1])
            if len(series) > 0
            else (0, 0, 0, 0, 0)
        )

    @staticmethod
    def _compute_controller_context(point: List[float]):
        return point

    # def _compute_controller_context(self, point: List[int] = None) -> List[float]:
    #     exact_execution_times: List[float] = list(
    #         compress(self.__time_dataset, self.__is_call_exact)
    #     )
    #     estimated_execution_times: List[float] = [
    #         x for x in self.__time_dataset if x not in exact_execution_times
    #     ]
    #     context = [
    #         self._estimator.get_error(),
    #         *self.__extract_time_context_features(exact_execution_times),
    #         *self.__extract_time_context_features(estimated_execution_times),
    #     ]
    #     if not point:
    #         return context
    #     else:
    #         return context + point

    @staticmethod
    def _compute_weighting_context(mab_loss: float) -> List[float]:
        global is_call_exact
        exact_calls = is_call_exact.count(True)
        # return [mab_loss, exact_calls, len(is_call_exact) - exact_calls]
        return [exact_calls, len(is_call_exact) - exact_calls]

    def _compute_estimated(
        self,
        point: List[int],
        mab: Optional[Tuple[MabHandler, Union[int, float]]] = None,
        mab_weight: MabHandler = None,
        is_point_in_context: bool = True,
    ) -> Tuple[List[float], float]:
        global is_call_exact
        is_call_exact.append(False)
        print("(#" + str(len(is_call_exact)) + ") Estimating Exact Call...")

        estimation, exec_time = Controller.measure_execution_time(
            self._estimator.predict,
            point,
        )
        self.__time_dataset.append(exec_time)
        self.__error_dataset.append(self._estimator.get_error())
        _, learn_time = Controller.measure_execution_time(
            self.learn,
            is_exact=False,
            point=point,
            exec_time=exec_time,
            mab=mab,
            mab_forced_probability=None,
            mab_weight=mab_weight,
            mab_weight_forced_probability=None,
            is_point_in_context=is_point_in_context,
        )

        Path(self.__controller_learn_time_debug).open("a").write(
            str(learn_time) + "," + str(1) + "\n"
        )

        return estimation, exec_time

    @abstractmethod
    def initialize_debug(self):
        pass

    @abstractmethod
    def write_debug(self, debug_info: Dict[str, Any]):
        pass

    def __reshape(self, to_reshape):
        if len(self.__time_dataset) > 1:
            return np.reshape(to_reshape, (-1, 1))
        else:
            return np.reshape(to_reshape, (1, -1))

    def get_time_error_z_score(
        self,
        time_weight: float = 1,
        error_weight: float = 1,
    ) -> float:
        reshaped_time = self.__reshape(self.__time_dataset)
        reshaped_error = self.__reshape(self.__error_dataset)
        self.__scaler.fit(reshaped_time)
        time_z_score: float = self.__scaler.transform(reshaped_time)[-1][0]
        self.__scaler.fit(reshaped_error)
        error_z_score: float = self.__scaler.transform(reshaped_error)[-1][0]
        out = (time_weight * time_z_score) + (error_weight * error_z_score)
        self.__cost_dataset.append(out)
        return out

    @staticmethod
    def __is_constant(x):
        return x.count(x[0]) == len(x)

    def get_time_error_correlation(self):
        try:
            samples = len(self.__error_dataset)
            independent_vars = 2
            if (
                self.__is_constant(self.__error_dataset)
                or self.__is_constant(self.__time_dataset)
                or self.__is_constant(self.__cost_dataset)
                or samples - independent_vars == 1
            ):
                return 0
            time_cost_correlation: float = sp.stats.pearsonr(
                self.__time_dataset, self.__cost_dataset
            )[0]
            error_cost_correlation: float = sp.stats.pearsonr(
                self.__error_dataset, self.__cost_dataset
            )[0]
            time_error_correlation: float = sp.stats.pearsonr(
                self.__time_dataset, self.__error_dataset
            )[0]
            R2 = np.sqrt(
                (
                    np.abs(time_cost_correlation ** 2)
                    + np.abs(error_cost_correlation ** 2)
                    - 2
                    * time_cost_correlation
                    * error_cost_correlation
                    * time_error_correlation
                )
                / (1 - np.abs(time_error_correlation))
            )
            if R2 == np.nan:
                return 0
            R2_adj = 1 - (
                ((1 - R2 ** 2) * (samples - 1)) / (samples - independent_vars - 1)
            )
            return -R2_adj
        except ValueError:
            return 0

    @staticmethod
    def __update_mean(old_mean: float, old_sample_number: int, new_sample: float):
        """
        Return updated mean in light of new observation without
        full recalculation.
        """
        return (old_sample_number * old_mean + new_sample) / (old_sample_number + 1)

    def compute_first_order_sensitivity_index(self):
        self.__time_model.learn_one(
            ChainedEstimator.X_to_river([self.__time_dataset[-1]]),
            self.__cost_dataset[-1],
        )
        self.__error_model.learn_one(
            ChainedEstimator.X_to_river([self.__error_dataset[-1]]),
            self.__cost_dataset[-1],
        )
        self.__fitted_time.append(
            self.__time_model.predict_one(
                ChainedEstimator.X_to_river([self.__time_dataset[-1]])
            )
        )
        self.__fitted_error.append(
            self.__error_model.predict_one(
                ChainedEstimator.X_to_river([self.__error_dataset[-1]])
            )
        )
        if len(self.__cost_dataset) == 1:
            return 0
        self.__fitted_time_mean = self.__update_mean(
            self.__fitted_time_mean, len(self.__fitted_time) - 1, self.__fitted_time[-1]
        )
        self.__fitted_error_mean = self.__update_mean(
            self.__fitted_error_mean,
            len(self.__fitted_error) - 1,
            self.__fitted_error[-1],
        )
        self.__cost_mean = self.__update_mean(
            self.__cost_mean, len(self.__cost_dataset) - 1, self.__cost_dataset[-1]
        )
        self.__time_importance += (
            self.__fitted_time[-1] - self.__fitted_time_mean
        ) ** 2 / (self.__cost_dataset[-1] - self.__cost_mean) ** 2
        self.__error_importance += (
            self.__fitted_error[-1] - self.__fitted_error_mean
        ) ** 2 / (self.__cost_dataset[-1] - self.__cost_mean) ** 2
        importance_sum = self.__time_importance + self.__error_importance
        time_importance = self.__time_importance / importance_sum
        error_importance = self.__error_importance / importance_sum
        return (0.5 - time_importance) ** 2 + (0.5 - error_importance) ** 2

    @staticmethod
    def measure_execution_time(
        func: Callable[..., Any], *args, **kwargs
    ) -> Tuple[Any, float]:
        start = time.time()
        out = func(*args, **kwargs)
        execution_time = time.time() - start
        return out, execution_time

    @abstractmethod
    def get_mab(self) -> MabHandler:
        pass

    @abstractmethod
    def get_weight_mab(self) -> MabHandler:
        pass

    @abstractmethod
    def get_parameters(self):
        pass

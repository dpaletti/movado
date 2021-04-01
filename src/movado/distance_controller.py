from collections import Callable
from typing import Union, Tuple, Optional, Dict
from movado.controller import Controller
from movado.estimator import Estimator
import numpy as np
from sklearn.neighbors import KDTree, DistanceMetric
from vowpalwabbit import pyvw
import time


class DistanceController(Controller):
    def __init__(
        self,
        exact_fitness: Callable["np.ndarray", Union[int, float]],
        estimator: Estimator,
        scaler="StandardScaler",
        nth_nearest: int = 3,
        distance_metric: str = "minkowski",
        threshold: str = "mean",
        **kwargs,
    ):
        super().__init__(scaler=scaler)
        self.__exact: Callable["np.ndarray", Union[int, float]] = exact_fitness
        self.__estimator: Estimator = estimator
        self.__dataset: "np.ndarray" = np.empty([0, 0])
        self.__distance_metric = distance_metric
        self.__nth_nearest = nth_nearest
        self.__are_neighbours_to_recompute: bool = True
        self.__knn: Optional[KDTree] = None
        self.__threshold: str = threshold
        self.__distance_metric_kwargs: Dict[str, Union[int, float]] = {
            k: kwargs[k] for k in kwargs.keys() if k.startswith("distance_")
        }
        self.__threshold_kwargs: Dict[str, Union[int, float]] = {
            k: kwargs[k] for k in kwargs.keys() if k.startswith("threshold_")
        }
        if self.__threshold == "fixed" and not self.__threshold_kwargs.get(
            "threshold_value"
        ):
            raise Exception(
                "Missing keyword argument threshold_value, this argument is mandatory"
                + " when using fixed threshold "
            )
        if self.__threshold == "mab":
            if not kwargs.get("mab_actions"):
                kwargs["mab_actions"] = 100
            if not kwargs.get("mab_bandwidth"):
                kwargs["mab_bandwidth"] = 5
            self.__mab = pyvw.vw(
                "--cats "
                + kwargs["mab_actions"]
                + "  --bandwidth "
                + kwargs["mab_bandwidth"]
                + " --min_value 0 --max_value 100 --chain_hash --coin --epsilon 0.2 -q :: "
            )

    def fitness(self, point: "np.ndarray") -> Union[int, float]:
        accuracy = 1
        if self.__get_nth_nearest_distance(point) < self.__get_threshold(point):
            estimation, exec_time = Controller.measure_execution_time(
                self.__estimator.estimate, point
            )
            if self.__threshold == "mab":
                self.__learn_mab(
                    self.get_time_error_z_score(
                        np.array(exec_time), self.__estimator.get_accuracy()
                    )
                )
            return estimation

        exact, exec_time = self.__exact(point)
        if self.__threshold == "mab":
            self.__learn_mab(
                self.get_time_error_z_score(np.array(exec_time), np.array(1))
            )
        return exact

    def add_point(self, point: "np.ndarray") -> None:
        np.append(self.__dataset, point, axis=0)
        self.__are_neighbours_to_recompute = True

    def __get_threshold(self, point: "np.ndarray") -> float:
        if self.__threshold == "mab":
            return self.__get_threshold_mab(point)
        return globals()["__get_threshold_" + self.__threshold]()

    def __get_threshold_fixed(self) -> float:
        return self.__threshold_kwargs["threshold_value"]

    def __get_threshold_mean(self):
        return np.mean(self.__get_distances())

    def __get_threshold_median(self):
        return np.median(self.__get_distances())

    def __get_distances(self) -> "np.ndarray":
        distances: "np.ndarray" = np.empty([0])
        for point in self.__dataset:
            distances = np.append(distances, self.__get_nth_nearest_distance(point))
        return distances

    def __get_nth_nearest_distance(self, point: "np.ndarray") -> float:
        if self.__are_neighbours_to_recompute:
            self.__knn = KDTree(
                self.__dataset,
                metric=DistanceMetric.get_metric(
                    self.__distance_metric, **self.__distance_metric_kwargs
                ),
            )
        return self.__knn.query([point], k=3)[0][0][self.__nth_nearest - 1]

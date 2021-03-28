from collections import Callable
from typing import Union, Tuple, Optional, Dict
from movado.controller import Controller
from movado.estimator import Estimator
import numpy as np
from sklearn.neighbors import KDTree, DistanceMetric
from vowpalwabbit import pyvw


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
        self.__distance_metric = distance_metric
        self.__nth_nearest = nth_nearest
        self.__last_dataset_size: int = 0
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
            self.__init_mab()

    def fitness(self, point: "np.ndarray") -> Union[int, float]:
        if self.__get_nth_nearest_distance(point) < self.__get_threshold():
            return self.__estimator.estimate(point)
        return self.__exact(point)

    def __get_threshold(self) -> float:
        return globals()["__get_threshold_" + self.__threshold]()

    def __get_threshold_fixed(self) -> float:
        return self.__threshold_kwargs["threshold_value"]

    def __get_threshold_mean(self):
        return np.mean(self.__get_distances())

    def __get_threshold_median(self):
        return np.median(self.__get_distances())

    def __init_mab(self):
        model = pyvw.vw("–cats", quiet=True)

    def __get_threshold_mab(self):
        pass

    def __get_distances(self) -> "np.ndarray":
        distances: "np.ndarray" = np.empty([0])
        for point in self.__estimator.get_dataset():
            distances = np.append(distances, self.__get_nth_nearest_distance(point))
        return distances

    def __get_nth_nearest_distance(self, point: "np.ndarray") -> float:
        current_size: int = self.__estimator.get_dataset().size
        if current_size != self.__last_dataset_size:
            self.__knn = KDTree(
                self.__estimator.get_dataset(),
                metric=DistanceMetric.get_metric(
                    self.__distance_metric, **self.__distance_metric_kwargs
                ),
            )
            self.__last_dataset_size = current_size
        return self.__knn.query([point], k=3)[0][0][self.__nth_nearest - 1]

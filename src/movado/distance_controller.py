from pathlib import Path
from typing import Union, Optional, Dict, List, Any
import numpy as np
from sklearn.neighbors import KDTree, DistanceMetric

from movado.controller import Controller
from movado.mab_handler_cats import MabHandlerCATS


class DistanceController(Controller):
    # do not annotate types in __init__, it throws "ABCMeta Object is not subscriptable"
    def __init__(
        self,
        exact_fitness,
        estimator,
        nth_nearest=3,
        distance_metric="minkowski",
        threshold="mab",
        debug=False,
        **kwargs,
    ):
        super().__init__(exact_fitness, estimator, debug)

        self.__evaluated_points: List[List[float]] = []
        self.__threshold: str = threshold
        self.__distance_metric = distance_metric
        self.__nth_nearest: int = nth_nearest
        self.__are_neighbours_to_recompute: bool = True
        self.__knn: Optional[KDTree] = None
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
            if kwargs.get("mab_epsilon"):
                self.__mab = MabHandlerCATS(
                    debug=debug, epsilon=kwargs.get("mab_epsilon")
                )
            else:
                self.__mab = MabHandlerCATS(debug=debug)
        if self._debug:
            Path(self._controller_debug).open("a").write(
                "Threshold, Nth_Nearest_Distance, Point, Exec_Time, Error, Estimation\n"
            )

    def compute_objective(self, point: List[int]) -> float:
        out: float
        mae: float
        self.__add_point(point)
        threshold = self.__get_threshold(point)
        nth_distance = self.__get_nth_nearest_distance(point)
        error = self._estimator.get_error()

        if nth_distance > threshold or error == 0.0:
            out, exec_time = self._compute_exact(
                point,
                (self.__mab, threshold) if self.__threshold == "mab" else None,
                mab_forced_probability=1
                if self.__threshold == "mab" and error == 0.0
                else None,
            )
        else:
            out, exec_time = self._compute_estimated(
                point, (self.__mab, threshold) if self.__threshold == "mab" else None
            )

        # TODO probably this check can be done only once
        if self._debug:
            self._write_debug(
                {
                    "Threshold": 0 if error == 0.0 else threshold,
                    "Nth_Nearest_Distance": 0
                    if not self.__evaluated_points
                    else nth_distance,
                    "Point": point,
                    "Exec_Time": exec_time,
                    "Error": error
                    if nth_distance > threshold or error == 0.0
                    else self._estimator.get_error(),
                    "Estimation": 0 if nth_distance > threshold or error == 0.0 else 1,
                }
            )
        return out

    def _write_debug(self, debug_info: Dict[str, Any]):
        Path(self._controller_debug).open("a").write(
            str(debug_info["Threshold"])
            + ", "
            + str(debug_info["Nth_Nearest_Distance"])
            + ", "
            + str(debug_info["Point"])
            + ", "
            + str(debug_info["Exec_Time"])
            + ", "
            + str(debug_info["Error"])
            + ", "
            + str(debug_info["Estimation"])
            + "\n"
        )

    def __add_point(self, point: List[float]) -> None:
        self.__evaluated_points.append(point)
        self.__are_neighbours_to_recompute = True

    def __get_threshold(self, point: List[int]) -> float:
        return getattr(self, "_get_threshold_" + self.__threshold)(
            point if self.__threshold == "mab" else None
        )

    def _get_threshold_fixed(self, *args) -> float:
        return self.__threshold_kwargs["threshold_value"]

    def _get_threshold_mean(self, *args):
        return np.mean(self._get_distances())

    def _get_threshold_median(self, *args):
        return np.median(self._get_distances())

    def _get_threshold_mab(self, point: List[int]) -> float:
        return self.__mab.predict(point) / 100

    def _get_distances(self) -> np.ndarray:
        distances: np.ndarray = np.empty([0])
        for point in self.__evaluated_points:
            distances = np.append(distances, self.__get_nth_nearest_distance(point))
        return distances

    def __get_nth_nearest_distance(self, point: List[float]) -> float:
        if self.__are_neighbours_to_recompute:
            self.__knn = KDTree(
                self.__evaluated_points,
                metric=DistanceMetric.get_metric(
                    self.__distance_metric, **self.__distance_metric_kwargs
                ),
            )
        k = min(self.__nth_nearest, len(self.__evaluated_points))
        return self.__knn.query([point], k=k)[0][0][k - 1]

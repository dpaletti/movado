from pathlib import Path
from typing import Union, Optional, Dict, List, Any
import numpy as np
from sklearn.neighbors import KDTree, DistanceMetric

from movado.controller import Controller
from movado.mab_handler_cats import MabHandlerCATS

# TODO make lists global
class DistanceController(Controller):
    # do not annotate types in __init__, it throws "ABCMeta Object is not subscriptable"
    def __init__(
        self,
        exact_fitness,
        estimator,
        self_exact: Optional[object] = None,
        problem_dimensionality: int = -1,
        solutions=None,
        nth_nearest=3,
        distance_metric="minkowski",
        threshold="mab",
        debug=False,
        skip_debug_initialization=False,
        mab_epsilon: float = 0.2,
        mab_bandwidth: int = 1,
        mab_weight: bool = True,
        mab_weight_epsilon: float = 0.2,
        mab_weight_bandwidth: int = 1,
    ):
        super().__init__(
            exact_fitness=exact_fitness,
            estimator=estimator,
            self_exact=self_exact,
            debug=debug,
        )

        self.__evaluated_points: List[List[float]] = []
        self.__threshold: str = threshold
        self.__distance_metric = distance_metric
        self.__nth_nearest: int = nth_nearest
        self.__are_neighbours_to_recompute: bool = True
        self.__knn: Optional[KDTree] = None
        self.__distances: List[float] = []
        self.__mab = None
        self.__weight_mab = None
        self.__params = (
            "Model_Parameters",
            {
                "nth_nearest": self.__nth_nearest,
                "mab_epsilon": mab_epsilon,
                "mab_bandwidth": mab_bandwidth,
                "mab_weight_epsilon": mab_weight_epsilon,
                "mab_weight_bandwidth": mab_weight_bandwidth,
            },
        )
        if self.__threshold == "mab":
            self.__mab = MabHandlerCATS(
                debug=debug,
                epsilon=mab_epsilon,
                bandwidth=mab_bandwidth,
                controller_params={self.__params[0]: self.__params[1]},
            )
            if mab_weight:
                self.__weight_mab = MabHandlerCATS(
                    debug=debug,
                    epsilon=mab_weight_epsilon,
                    bandwidth=mab_weight_bandwidth,
                    debug_path="mab_weight",
                    controller_params={self.__params[0]: self.__params[1]},
                )

        if self._debug and not skip_debug_initialization:
            self.initialize_debug()

    def initialize_debug(self):
        Path(self._controller_debug).open("a").write(
            "Model_Parameters, Threshold, Nth_Nearest_Distance, Point, Exec_Time, Error, Estimation\n"
        )

    def compute_objective(self, point: List[int]) -> List[float]:
        out: List[float]
        mae: float
        self.__add_point(point)
        threshold = self.__get_threshold(point)
        nth_distance = self.__get_nth_nearest_distance(point)
        error = self._estimator.get_error()

        if nth_distance > threshold or error == 0.0:
            out, exec_time = self._compute_exact(
                point,
                (self.__mab, threshold * 100) if self.__threshold == "mab" else None,
                1 if self.__threshold == "mab" and error == 0.0 else None,
                self.__weight_mab
                if self.__threshold == "mab" and self.__weight_mab
                else None,
                1
                if self.__threshold == "mab" and error == 0.0 and self.__weight_mab
                else None,
            )
        else:
            out, exec_time = self._compute_estimated(
                point,
                (self.__mab, threshold * 100) if self.__threshold == "mab" else None,
                self.__weight_mab
                if self.__threshold == "mab" and self.__weight_mab
                else None,
            )

        # TODO probably this check can be done only once
        if self._debug:
            self.write_debug(
                {
                    self.__params[0]: self.__params[1],
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

    def write_debug(self, debug_info: Dict[str, Any]):
        Path(self._controller_debug).open("a").write(
            str(debug_info["Model_Parameters"])
            + ", "
            + str(debug_info["Threshold"])
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
                metric=DistanceMetric.get_metric(self.__distance_metric),
            )
        k = min(self.__nth_nearest, len(self.__evaluated_points))
        distance = self.__knn.query([point], k=k)[0][0][k - 1]
        self.__distances.append(distance)
        diameter = max(self.__distances) - min(self.__distances)
        if diameter == 0:
            return 0
        else:
            return (distance - min(self.__distances)) / diameter

    def get_mean_cost(self) -> float:
        if self.__threshold != "mab":
            raise Exception("Mean cost is defined only for Controllers employing Mabs")
        return self.__mab.get_mean_cost()

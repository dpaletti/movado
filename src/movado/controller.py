from abc import abstractmethod, ABC
from typing import Tuple, Optional, Callable, Any, List, Dict, Union
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from pathlib import Path

from movado.estimator import Estimator
from movado.mab_handler import MabHandler


class Controller(ABC):
    @abstractmethod
    def __init__(
        self,
        exact_fitness: Callable[[List[float]], List[float]],
        estimator: Estimator,
        debug=False,
    ):
        self.__scaler = StandardScaler()
        self.__time_dataset: List[float] = []
        self.__error_dataset: List[float] = []
        self.__to_reshape = False
        self._exact: Callable[[List[float]], List[float]] = exact_fitness
        self._estimator: Estimator = estimator
        self._debug = debug
        if self._debug:
            Path("movado_debug").mkdir(exist_ok=True)
            self._controller_debug = "movado_debug/controller.csv"
            Path(self._controller_debug).open("w").close()

    @abstractmethod
    def compute_objective(self, point: List[int]) -> float:
        pass

    def _compute_exact(
        self,
        point: List[int],
        mab: Optional[Tuple[MabHandler, Union[int, float]]] = None,
        mab_forced_probability: Optional[int] = None,
    ) -> Tuple[List[float], float]:
        exact, exec_time = Controller.measure_execution_time(self._exact, point)
        self._estimator.train(point, exact)
        if mab:
            mab[0].learn(
                mab[1], self.get_time_error_z_score(exec_time, 0), point
            ), mab_forced_probability
        return exact, exec_time

    def _compute_estimated(
        self,
        point: List[int],
        mab: Optional[Tuple[MabHandler, Union[int, float]]] = None,
    ) -> Tuple[List[float], float]:
        estimation, exec_time = Controller.measure_execution_time(
            self._estimator.predict, point
        )
        if mab:
            mab[0].learn(
                mab[1],
                self.get_time_error_z_score(exec_time, self._estimator.get_error()),
                point,
            )
        return estimation, exec_time

    @abstractmethod
    def _write_debug(self, debug_info: Dict[str, Any]):
        pass

    def get_time_error_z_score(self, response_time: float, error: float) -> float:
        self.__time_dataset.append(response_time)
        self.__error_dataset.append(error)
        if len(self.__time_dataset) > 1:
            reshaped_time = np.reshape(self.__time_dataset, (-1, 1))
            reshaped_error = np.reshape(self.__error_dataset, (-1, 1))
        else:
            reshaped_time = np.reshape(self.__time_dataset, (1, -1))
            reshaped_error = np.reshape(self.__error_dataset, (1, -1))
        self.__scaler.fit(reshaped_time)
        time_z_score: float = self.__scaler.transform(reshaped_time)[-1][0]
        self.__scaler.fit(reshaped_error)
        error_z_score: float = self.__scaler.transform(reshaped_error)[-1][0]
        return -time_z_score - error_z_score

    @staticmethod
    def measure_execution_time(
        func: Callable[..., Any], *args, **kwargs
    ) -> Tuple[Any, float]:
        start = time.time()
        out = func(*args, **kwargs)
        execution_time = time.time() - start
        return out, execution_time

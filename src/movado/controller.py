import abc
from abc import abstractmethod
from typing import Union, Tuple, Optional, Callable, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import time


class Controller(abc.ABC):
    @abstractmethod
    def __init__(self, scaler="StandardScaler"):
        self.__scaler: Union[StandardScaler, RobustScaler] = (
            StandardScaler() if scaler.lower() == "standardscaler" else RobustScaler
        )
        self.__time_dataset: Optional["np.ndarray"] = np.ndarray([0])
        self.__error_dataset: Optional["np.ndarray"] = np.ndarray([0])

    @abstractmethod
    def fitness(self, point: [Union[Tuple[int], Tuple[float]]]) -> Union[int, float]:
        pass

    def get_time_error_z_score(
        self, response_time: "np.ndarray", accuracy: "np.ndarray"
    ) -> float:
        self.__time_dataset = np.append(self.__time_dataset, response_time)
        self.__error_dataset = np.append(self.__error_dataset, 1 / accuracy)
        self.__scaler.fit(self.__time_dataset)
        time_z_score: float = self.__scaler.transform(self.__time_dataset)[-1][0]
        self.__scaler.fit(self.__error_dataset)
        error_z_score: float = self.__scaler.transform(self.__error_dataset)[-1][0]
        return -time_z_score - error_z_score

    @staticmethod
    def measure_execution_time(
        func: Callable[[Any, ...], Any], *args, **kwargs
    ) -> Tuple[Any, float]:
        start = time.process_time()
        out = func(*args, **kwargs)
        execution_time = time.process_time() - start
        return out, execution_time

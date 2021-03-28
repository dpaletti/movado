import abc
from abc import abstractmethod
from typing import Union, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np


class Controller(abc.ABC):
    default: str = "DistanceController"

    @abstractmethod
    def __init__(self, scaler="StandardScaler"):
        self.__scaler: Union[StandardScaler, RobustScaler] = (
            StandardScaler() if scaler.lower() == "standardscaler" else RobustScaler
        )
        self.__time_dataset: Optional["np.ndarray"] = np.ndarray([0])
        self.__acc_dataset: Optional["np.ndarray"] = np.ndarray([0])

    @abstractmethod
    def fitness(self, point: [Union[Tuple[int], Tuple[float]]]) -> Union[int, float]:
        pass

    def get_time_acc_z_score(
        self, time: "np.ndarray", accuracy: "np.ndarray"
    ) -> Tuple[float, float]:
        self.__time_dataset = np.append(self.__time_dataset, time)
        self.__acc_dataset = np.append(self.__acc_dataset, accuracy)
        self.__scaler.fit(self.__time_dataset)
        time_z_score: float = self.__scaler.transform(self.__time_dataset)[-1][0]
        self.__scaler.fit(self.__acc_dataset)
        accuracy_z_score: float = self.__scaler.transform(self.__acc_dataset)[-1][0]
        return time_z_score, accuracy_z_score

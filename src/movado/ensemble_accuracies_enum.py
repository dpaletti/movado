from enum import Enum


class EnsembleAccuracies(Enum):
    MIN = "max"
    MAX = "min"
    AVG = "np.avg"

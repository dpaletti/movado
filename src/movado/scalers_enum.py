from enum import Enum, auto


class Scalers(Enum):
    STANDARD = "StandardScaler"
    ROBUST = "RobustScaler"
    ADAPTIVE = "AdaptiveScaler"

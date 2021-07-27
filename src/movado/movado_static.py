from typing import List

# noinspection PyUnresolvedReferences
from movado.mab_controller import MabController  # pylint: disable=unused-import

# noinspection PyUnresolvedReferences
from movado.voting_controller import VotingController  # pylint: disable=unused-import

# noinspection PyUnresolvedReferences
from movado.shadow_controller import ShadowController  # pylint: disable=unused-import

# noinspection PyUnresolvedReferences
from movado.distance_controller import (
    DistanceController,
)  # pylint: disable=unused-import

# noinspection PyUnresolvedReferences
from movado.kernel_regression_model import (
    KernelRegressionModel,
)  # pylint: disable=unused-import

# noinspection PyUnresolvedReferences
from movado.hoeffding_adaptive_tree_model import (
    HoeffdingAdaptiveTreeModel,
)  # pylint: disable=unused-import

models: List[str] = [
    model.replace("Model", "")
    for model in globals().keys()
    if ("Model" in model) and len(model) > 5
]
controllers: List[str] = [
    ctrl.replace("Controller", "")
    for ctrl in globals().keys()
    if ("Controller" in ctrl) and len(ctrl) > 10
]

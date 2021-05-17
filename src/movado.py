from typing import List, Callable

from controller import Controller
from estimator import Estimator
from hoeffding_adaptive_tree_chained_estimator import (
    HoeffdingAdaptiveTreeChainedEstimator,
)
from functools import wraps
from distance_controller import DistanceController
from pydoc import locate

# This imports are unused but populate the global symbol table
from mab_controller import MabController


def approximate(**kwargs):
    def approximate_decorator(func: Callable[[List[float]], float]):
        controller: Controller
        estimator: Estimator
        is_first_call: bool = True

        @wraps(func)
        def wrapper(point):
            nonlocal is_first_call
            nonlocal controller
            nonlocal estimator
            if is_first_call:
                is_first_call = False
                selected_estimator = globals().get(
                    str(kwargs.get("estimator")) + "Estimator"
                )
                print(globals().keys())
                estimator = (
                    HoeffdingAdaptiveTreeChainedEstimator
                    if not selected_estimator
                    else selected_estimator
                )
                estimator = estimator(
                    **{
                        k[k.find("_") + 1 :]: v
                        for k, v in kwargs.items()
                        if k[: k.find("_")] == "estimator"
                    }
                )
                selected_controller = globals().get(
                    str(kwargs.get("controller")) + "Controller"
                )
                controller = (
                    DistanceController
                    if not selected_controller
                    else selected_controller
                )
                controller = controller(
                    func,
                    estimator,
                    **{
                        k[k.find("_") + 1 :]: v
                        for k, v in kwargs.items()
                        if k[: k.find("_")] == "controller"
                    }
                )

            controller.compute_objective(
                point
            )  # we expect the first and only argument of the function to be the input point

        return wrapper

    return approximate_decorator

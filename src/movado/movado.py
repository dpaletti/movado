from typing import List, Callable

from movado.controller import Controller
from movado.estimator import Estimator
from functools import wraps
from movado.distance_controller import DistanceController
from movado.hoeffding_adaptive_tree_model import HoeffdingAdaptiveTreeModel
from movado.chained_estimator import ChainedEstimator

# This imports are unused but populate the global symbol table for the globals() call
from movado.mab_controller import MabController
from movado.kernel_regression_model import KernelRegressionModel


def approximate(**kwargs):
    def approximate_decorator(func: Callable[[List[float]], List[float]]):
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
                selected_model = globals().get(str(kwargs.get("estimator")) + "Model")
                model = (
                    HoeffdingAdaptiveTreeModel if not selected_model else selected_model
                )
                # TODO add multi-output non-chained estimators
                estimator = ChainedEstimator(
                    model(
                        **{
                            k[k.find("_") + 1 :]: v
                            for k, v in kwargs.items()
                            if k[: k.find("_")] == "estimator"
                        }
                    )
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


class Movado:
    def __init__(self):
        self.models = [model for model in globals().keys() if "Model" in model]
        self.controllers = [
            controller for controller in globals().keys() if "Controller" in controller
        ]

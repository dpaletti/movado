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


def approximate(
    **kwargs,
) -> Callable[
    [Callable[[List[float]], List[float]]], Callable[[List[float]], List[float]]
]:
    def approximate_decorator(
        func: Callable[[List[float]], List[float]]
    ) -> Callable[[List[float]], List[float]]:
        controller: Controller
        estimator: Estimator
        is_first_call: bool = True

        @wraps(func)
        def wrapper(*wrapper_args) -> List[float]:
            # args must either contain the point to be analyzed
            # or it must contain self (for class methods) and then the point
            nonlocal is_first_call
            nonlocal controller
            nonlocal estimator
            if len(wrapper_args) < 1 or len(wrapper_args) > 2:
                raise Exception(
                    "The decorated function must have a single input which is a list of numbers, "
                    + "self may also be present for non-static class methods, in such case self is "
                    + "expected to be the first argument followed by the point and nothing"
                )
            if len(wrapper_args) == 1:
                point = wrapper_args[0]
            else:
                point = wrapper_args[1]
            if is_first_call:
                is_first_call = False
                selected_model = globals().get(str(kwargs.get("estimator")) + "Model")
                model = (
                    HoeffdingAdaptiveTreeModel if not selected_model else selected_model
                )
                # TODO add multi-output non-chained estimators
                outputs: int = kwargs.get("outputs")
                if not outputs:
                    raise Exception(
                        "Please specify outputs as a kwarg as the number of targets of the fitness function"
                    )
                estimator = ChainedEstimator(
                    model(
                        **{
                            k[k.find("_") + 1 :]: v
                            for k, v in kwargs.items()
                            if k[: k.find("_")] == "estimator"
                        }
                    ),
                    outputs,
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
                    self_exact=None if len(wrapper_args) == 1 else wrapper_args[0],
                    **{
                        k[k.find("_") + 1 :]: v
                        for k, v in kwargs.items()
                        if k[: k.find("_")] == "controller"
                    },
                )

            return list(
                controller.compute_objective(point)
            )  # we expect the first and only argument of the function to be the input point

        if kwargs.get("disabled"):
            return func
        return wrapper

    return approximate_decorator


class Movado:
    def __init__(self):
        self.models = [
            model.replace("Model", "")
            for model in globals().keys()
            if ("Model" in model) and len(model) > 5
        ]
        self.controllers = [
            controller.replace("Controller", "")
            for controller in globals().keys()
            if ("Controller" in controller) and len(controller) > 10
        ]

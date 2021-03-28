from typing import Union, Iterator, ClassVar
import numpy as np

from movado.controller import Controller
from movado.estimator import Estimator


def approximate(fitness: ["np.ndarray", Union[int, float]]):
    controller: Controller
    estimator: Estimator
    is_first_call: bool = True

    def get_class(parent: ClassVar, class_name: str) -> ClassVar:
        if class_name == "Controller":
            class_name = parent.default
        child_names_list: Iterator = map(
            lambda sc: sc.__name(), parent.__subclasses__()
        )

        for i, cn in enumerate(child_names_list):
            if cn == class_name:
                return parent.__subclasses__()[i]

    def approximate_decorator(*args, **kwargs):
        nonlocal is_first_call
        nonlocal controller
        nonlocal estimator
        if is_first_call:
            if args:
                print("Non-keyword arguments are ignored")
            controller = get_class(
                Controller, kwargs.get("controller").lower() + "Controller"
            )(fitness, estimator)
            estimator = get_class(Estimator, kwargs.get("estimator"))()
            is_first_call = False
        else:
            controller.fitness()

from time import sleep
from typing import List

from movado import approximate
from random import sample, random
import numpy as np

sleep_time = 5


class TestClass:
    @approximate(outputs=3, controller="Shadow", controller_debug=True, voters=-1)
    def sleep_func(self, point: List[float]) -> List[float]:
        global sleep_time
        x = np.linspace(-np.pi, np.pi, 201)
        sleep(sleep_time)
        out = [
            sample(list(np.sin(x) + np.random.normal(0, 1, 201)), 1)[0] + p
            for p in point
        ]
        return out


def test_movado_short_sleep():
    test_class = TestClass()
    for i in range(0, 100):
        point = [random(), random(), random()]
        test_class.sleep_func(point)


test_movado_short_sleep()

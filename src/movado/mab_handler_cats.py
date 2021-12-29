from typing import List, Tuple

from vowpalwabbit import pyvw

from movado.mab_handler import MabHandler


class MabHandlerCATS(MabHandler):
    def __init__(
        self,
        bandwidth: int = 1,
        epsilon: float = 0.2,
        debug=False,
        controller_params: dict = None,
        debug_path: str = "mab",
        skip_debug_initialization: bool = False,
    ):
        """
        CATS parameters interpretation:
        - the range of outputs is (min_value, max_value), in our case (0, 100)
        - mab_actions determines the number of buckets (discretization) we apply to the range,
          in our case range is actions is 100 thus we have 100 buckets of length 1 ((0, 1], [1, 2], ..., [99, 100))
        - bandwidth determines the density function of the outputs for each bucket outputs
          will be more dense around [bucket_center+bandwidth, bucket_center-bandwidth]
          where bucket_center = (bucket_min + bucket_max)/2. In our case bandwidth defaults to 1.
        """
        super(MabHandlerCATS, self).__init__(
            debug,
            debug_path,
            controller_params=controller_params,
            skip_debug_initialization=skip_debug_initialization,
        )

        self._sample_prefix = "ca "
        if bandwidth < 0:
            raise Exception(
                "Invalid bandwidth value: "
                + str(bandwidth)
                + " it must be greater or equal than 1"
            )
        if epsilon < 0 or epsilon > 1:
            raise Exception(
                "Invalid epsilon value: " + str(epsilon) + "it must be between 0 and 1"
            )
        mab_actions: int = 100
        self._mab: pyvw.vw = pyvw.vw(
            "--cats "
            + str(mab_actions)
            + "  --bandwidth "
            + str(bandwidth)
            + " --min_value 0 --max_value 100 --chain_hash --coin --epsilon "
            + str(epsilon)
            + " --quiet"
            + " --random_seed 0"
        )

    def predict(self, context: List[float]) -> float:
        context_str: str = "| "
        for feature in context:
            context_str += str(feature) + " "
        context_str.strip()
        action: Tuple[float, float] = self._mab.predict(context_str)
        self._last_action = action[0]
        self._last_predict_probability = action[1]
        return action[0]

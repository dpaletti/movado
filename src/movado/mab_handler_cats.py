from typing import List, Tuple

from vowpalwabbit import pyvw

from movado.mab_handler import MabHandler


class MabHandlerCATS(MabHandler):
    def __init__(
        self,
        mab_bandwidth: int = 1,
        epsilon: float = 0.2,
        debug=False,
        debug_path: str = "mab",
    ):
        super(MabHandlerCATS, self).__init__(debug, debug_path)
        self._sample_prefix = "ca "
        if mab_bandwidth < 0:
            raise Exception(
                "Invalid bandwidth value: "
                + str(mab_bandwidth)
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
            + str(mab_bandwidth)
            + " --min_value 0 --max_value 100 --chain_hash --coin --epsilon 0.2 -q ::"
        )

    def predict(self, context: List[float]) -> float:
        context_str: str = "| "
        for feature in context:
            context_str += str(feature) + " "
        context_str.strip()
        action: Tuple[float, float] = self._mab.predict(context_str)
        self._last_predict_probability = action[1]
        return action[0]

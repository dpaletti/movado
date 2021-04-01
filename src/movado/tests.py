from vowpalwabbit import pyvw
import timeit

def retx(x):
    return x

num_actions = 100
bandwidth = 5

vw = pyvw.vw("--cats " + str(num_actions) + "  --bandwidth " + str(bandwidth) + " --min_value 0 --max_value 100 --chain_hash --coin --epsilon 0.2 -q :: ")

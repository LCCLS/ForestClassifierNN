import math


def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.001
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

import math
import numpy as np
from typing import Union

from minitorch.numeric_precision import *

dtype = np.int8


def _to_int8(x: Union[float, np.int8]) -> np.int8:
    # Clip to int8 range [-128, 127]
    return np.int8(max(min(round(float(x)), 127), -128))


def mul(x: np.int8, y: np.int8) -> np.int8:
    return _to_int8(int(x) * int(y))


def add(x: np.int8, y: np.int8) -> np.int8:
    return _to_int8(int(x) + int(y))


def neg(x: np.int8) -> np.int8:
    return _to_int8(-int(x))


def sigmoid(x: np.int8) -> np.int8:
    # Scale for int8: map [-128, 127] to [0, 1] * 127
    fx = float(x) / 128.0
    sig = (
        1.0 / (1.0 + math.exp(-fx)) if fx >= 0 else math.exp(fx) / (1.0 + math.exp(fx))
    )
    return _to_int8(sig * 127)


def relu(x: np.int8) -> np.int8:
    return x if x > 0 else np.int8(0)


def log(x: np.int8) -> np.int8:
    # Scale for int8: map [1, 127] to [0, ln(127)] * 127
    if x <= 0:
        return np.int8(-128)  # Represent negative infinity
    return _to_int8(math.log(float(x)) * 32)


def exp(x: np.int8) -> np.int8:
    # Scale for int8: map [-128, 127] to [e^-1, e^1] * 127
    fx = float(x) / 127.0
    return _to_int8(math.exp(fx) * 64)


def inv(x: np.int8) -> np.int8:
    # Scale for int8: map [-128, 127] to [-1, 1] * 127
    if x == 0:
        return np.int8(127)  # Represent infinity
    return _to_int8((127.0 / float(x)))


def is_close(x: np.int8, y: np.int8) -> bool:
    return abs(int(x) - int(y)) <= 1

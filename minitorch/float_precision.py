import math
import numpy as np
from .numeric_precision import *

dtype = float
encode = float
decode = float
ONE = 1.0
ZERO = 0.0
RTOL = 1e-2
ATOL = 1e-2
EPSILON = math.ulp(0.5)


def id(x: float) -> float:
    return x


def mul(x: float, y: float) -> float:
    return x * y


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    return float(x) if x > 0 else 0.0


def relu_back(x: float, y: float) -> float:
    return float(y) if x > 0 else 0.0


def log(x: float) -> float:
    return math.log(x)


def log_back(x: float, y: float) -> float:
    return y / x


def exp(x: float) -> float:
    return math.exp(x)


def exp_back(x: float, y: float) -> float:
    return math.exp(x) * y


def inv(x: float) -> float:
    return 1.0 / x


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid_back(x: float, y: float) -> float:
    return y * (1.0 - y)


def inv_back(x: float, y: float) -> float:
    return -y / (x * x)


def max_(x: float, y: float) -> float:
    return float(x) if x > y else float(y)


def min_(x: float, y: float) -> float:
    return x if x <= y else y


def all_(x: float, y: float) -> float:
    if x == 0 or y == 0:
        return float(0)
    return float(1)


def any_(x: float, y: float) -> float:
    if x == 0 and y == 0:
        return float(0)
    return float(1)


def lt(x: float, y: float) -> bool:
    return x < y


def le(x: float, y: float) -> bool:
    return x <= y


def eq(x: float, y: float) -> bool:
    return x == y


def map(fn: Callable[[float], float], l: Iterable[float]) -> Iterable[float]:
    return [fn(x) for x in l]


def zipWith(
    fn: Callable[[float, float], float], l1: Iterable[float], l2: Iterable[float]
) -> Iterable[float]:
    return [fn(x, y) for x, y in zip(l1, l2)]


def reduce(
    fn: Callable[[float, float], float], l: Iterable[float], init: float
) -> float:
    return init if not l else reduce(fn, l[1:], fn(init, l[0]))  # type: ignore


def sum(l: Iterable[float]) -> float:
    return reduce(add, l, 0.0)


def prod(l: Iterable[float]) -> float:
    return reduce(mul, l, 1.0)


def zeros(size: int, dtype: type = float) -> np.ndarray:
    return np.zeros(size, dtype=dtype)


def rand(size: int, dtype: type = float) -> np.ndarray:
    if dtype != float:
        raise ValueError("Only float supported")
    return np.random.rand(size)

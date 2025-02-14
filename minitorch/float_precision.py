from .numeric_precision import *

dtype = float


def id(x: float) -> float:
    return x


def mul(x: float, y: float) -> float:
    return float(x * y)


def add(x: float, y: float) -> float:
    return float(x + y)


def neg(x: float) -> float:
    return -float(x)


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


def max(x: float, y: float) -> float:
    return float(x) if x > y else float(y)


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y

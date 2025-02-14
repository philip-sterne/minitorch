import math
import numpy as np
from typing import TypeVar, Generic, Union, Type, Callable, Iterable

T = TypeVar("T", float, np.int8)
# This file doesn't do anything useful, except define all
# the functions that need to be implemented for each precision.
# You can use this file to see what functions need to be implemented,
# but you will not need to edit this file.


type: Type[T]


def mul(x: T, y: T) -> T:
    raise NotImplementedError


def id(x: T) -> T:
    raise NotImplementedError


def add(x: T, y: T) -> T:
    raise NotImplementedError


def neg(x: T) -> T:
    raise NotImplementedError


def sigmoid(x: T) -> T:
    raise NotImplementedError


def sigmoid_back(x: T, y: T) -> T:
    raise NotImplementedError


def relu(x: T) -> T:
    raise NotImplementedError


def relu_back(x: T, y: T) -> T:
    raise NotImplementedError


def log(x: T) -> T:
    raise NotImplementedError


def log_back(x: T, y: T) -> T:
    raise NotImplementedError


def exp(x: T) -> T:
    raise NotImplementedError


def exp_back(x: T, y: T) -> T:
    raise NotImplementedError


def inv(x: T) -> T:
    raise NotImplementedError


def inv_back(x: T, y: T) -> T:
    raise NotImplementedError


def is_close(x: T, y: T) -> bool:
    raise NotImplementedError


def lt(x: T, y: T) -> bool:
    raise NotImplementedError


def eq(x: T, y: T) -> bool:
    raise NotImplementedError


def max(x: T, y: T) -> T:
    return x if x > y else y


def map(fn: Callable[[T], T], l: Iterable[T]) -> Iterable[T]:
    return [fn(x) for x in l]


def zipWith(fn: Callable[[T, T], T], l1: Iterable[T], l2: Iterable[T]) -> Iterable[T]:
    return [fn(x, y) for x, y in zip(l1, l2)]


def reduce(fn: Callable[[T, T], T], l: Iterable[T], init: T) -> T:
    return init if not l else reduce(fn, l[1:], fn(init, l[0]))


def sum(l: Iterable[T]) -> T:
    return reduce(add, l, T(0))


def prod(l: Iterable[T]) -> T:
    return reduce(mul, l, T(1))

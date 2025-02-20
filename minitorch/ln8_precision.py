import math
import numpy as np
from typing import Callable, Iterable
from .numeric_precision import *

# Precompute the tables used for log‑based addition.
def _build_plus_table() -> list[int]:
    """
    plus_table[delta] = integer in [0..8] approximating:
      8 * log2(1 + 2^(-delta/8))
    """
    tbl = [0] * 128
    for delta in range(128):
        val = math.log2(1.0 + 2.0 ** (-delta / 8.0))
        off = round(8.0 * val)
        off = int(max(0, min(off, 8)))
        tbl[delta] = off
    return tbl

def _build_minus_table() -> list[int]:
    """
    minus_table[delta] = integer in [0..8] approximating:
      8 * log2(|1 - 2^(-delta/8)|)
    (with delta=0 giving an offset of 0)
    """
    tbl = [0] * 128
    for delta in range(128):
        x = 2.0 ** (-delta / 8.0)
        fraction = abs(1.0 - x)
        if fraction < 1e-15:
            tbl[delta] = 0
        else:
            val = math.log2(fraction)
            off = round(8.0 * val)
            tbl[delta] = off
    return tbl

PLUS_TABLE = _build_plus_table()
MINUS_TABLE = _build_minus_table()

#-----------------------------------------------------------
# Conversion functions between a Python float and our FP8 code.
# The representation is:
#    if code >= 0:   value = + 2^((code - 64)/8)
#    if code < 0:    value = - 2^(((-code) - 64)/8)
# with a special case that code==0 decodes to ~1/256.
#-----------------------------------------------------------
def decode(fp8: np.int8) -> float:
    """Convert an 8-bit code to a float."""
    x_int = int(fp8)
    if x_int == 0:
        return 1.0 / 256.0
    elif x_int > 0:
        return math.pow(2.0, (x_int - 64) / 8.0)
    else:
        return -math.pow(2.0, ((-x_int) - 64) / 8.0)

def encode(value: float) -> np.int8:
    """
    Convert a float to an 8-bit code.
    For value==0, we return 0.
    For nonzero value, we compute:
         code = round(64 + 8 * log2(|value|))
    and then apply the sign.
    
    Positive numbers are clamped to a maximum code of 127.
    Negative numbers are clamped to a minimum code of -128.
    """
    if value == 0.0:
        return np.int8(0)
    sign = 1 if value >= 0 else -1
    mag = abs(value)
    # If the magnitude is extremely small, clamp it.
    if mag < 1e-3:
        return np.int8(0 if sign > 0 else -1)
    p = math.log2(mag)
    p = max(-1024, min(p, 1024))
    p_approx = round(64 + 8 * p)
    if sign > 0:
        # Positive codes: allowed range is 0 ... 127
        p_approx = max(0, min(p_approx, 127))
    else:
        # Negative codes: allowed range is -1 ... -128 (i.e. absolute value up to 128)
        p_approx = max(0, min(p_approx, 128))
    return np.int8(p_approx if sign > 0 else -p_approx)

#-----------------------------------------------------------
# Arithmetic Operations
#-----------------------------------------------------------
def add(x: np.int8, y: np.int8) -> np.int8:
    """
    Add two FP8 numbers (encoded as np.int8) using the log‐based method.
    If both operands have the same sign, we add an offset from PLUS_TABLE;
    if they differ, we subtract using MINUS_TABLE.
    """
    if x.size != 1:
        import pdb; pdb.set_trace()
    if y.size != 1:
        import pdb; pdb.set_trace()
    x_int = int(x)
    y_int = int(y)
    s1 = 1 if x_int >= 0 else -1
    s2 = 1 if y_int >= 0 else -1
    a1 = x_int if x_int >= 0 else -x_int
    a2 = y_int if y_int >= 0 else -y_int

    if s1 == s2:
        # Same sign: use plus table.
        if a1 >= a2:
            big, small = a1, a2
        else:
            big, small = a2, a1
        delta = big - small
        delta = int(max(0, min(delta, 127)))
        off = PLUS_TABLE[delta]
        sum_exp = big + off
        sum_exp = max(0, min(sum_exp, 127))
        result = sum_exp if s1 > 0 else -sum_exp
        return np.int8(result)
    else:
        # Different signs: subtract the smaller magnitude from the larger.
        if a1 > a2:
            big, small, res_sign = a1, a2, s1
        elif a2 > a1:
            big, small, res_sign = a2, a1, s2
        else:
            return np.int8(0)  # Equal magnitudes cancel.
        delta = big - small
        delta = int(max(0, min(delta, 127)))
        off = MINUS_TABLE[delta]
        diff_exp = big + off
        diff_exp = max(0, min(diff_exp, 127))
        result = diff_exp if res_sign > 0 else -diff_exp
        return np.int8(result)

def neg(x: np.int8) -> np.int8:
    """Negate an FP8 number."""
    result = -int(x)
    result = max(-128, min(result, 127))
    return np.int8(result)

def mul(x: np.int8, y: np.int8) -> np.int8:
    """
    Multiply two FP8 numbers.
    The multiplication rule is based on adding the exponents:
         p = (|x| - 64) + (|y| - 64) + 64 = |x| + |y| - 64
    with the sign given by the product of the signs.
    """
    x_int = int(x)
    y_int = int(y)
    s1 = 1 if x_int >= 0 else -1
    s2 = 1 if y_int >= 0 else -1
    a1 = x_int if x_int >= 0 else -x_int
    a2 = y_int if y_int >= 0 else -y_int
    p = a1 + a2 - 64
    p = max(0, min(p, 127))
    result = p if (s1 * s2 > 0) else -p
    result = max(-128, min(result, 127))
    return np.int8(result)

def truediv(x: np.int8, y: np.int8) -> np.int8:
    """
    Divide two FP8 numbers.
    The division rule is based on subtracting the exponents:
         p = (|x| - 64) - (|y| - 64) + 64 = |x| - |y| + 64
    with the sign given by the product of the signs.
    """
    x_int = int(x)
    y_int = int(y)
    s1 = 1 if x_int >= 0 else -1
    s2 = 1 if y_int >= 0 else -1
    a1 = x_int if x_int >= 0 else -x_int
    a2 = y_int if y_int >= 0 else -y_int
    p = a1 - a2 + 64
    p = max(0, min(p, 127))
    result = p if (s1 * s2 > 0) else -p
    result = max(-128, min(result, 127))
    return np.int8(result)

def id(x: np.int8) -> np.int8:
    """Identity function."""
    return x

#-----------------------------------------------------------
# Nonlinear and Activation Functions
#-----------------------------------------------------------
def sigmoid(x: np.int8) -> np.int8:
    """
    Compute the sigmoid function.
       sigmoid(f) = 1/(1+exp(-f))
    where f is the decoded FP8 number.
    """
    x_float = decode(x)
    s = 1.0 / (1.0 + math.exp(-x_float))
    return encode(s)

def sigmoid_back(x: np.int8, y: np.int8) -> np.int8:
    """
    Backward pass for sigmoid.
    Given upstream gradient y, the local derivative is
       sigmoid(x) * (1 - sigmoid(x))
    """
    x_float = decode(x)
    s = 1.0 / (1.0 + math.exp(-x_float))
    deriv = s * (1.0 - s)
    y_float = decode(y)
    grad = y_float * deriv
    return encode(grad)

def relu(x: np.int8) -> np.int8:
    """
    ReLU activation: if x is negative, return FP8(0)
    (which decodes to ~1/256); otherwise return x.
    """
    return x if int(x) >= 0 else np.int8(0)

def relu_back(x: np.int8, y: np.int8) -> np.int8:
    """
    Backward pass for ReLU: multiply the upstream gradient y
    by 1 if x >= 0 and 0 if x < 0.
    """
    deriv = 1.0 if int(x) >= 0 else 0.0
    y_float = decode(y)
    grad = y_float * deriv
    return encode(grad)

def log(x: np.int8) -> np.int8:
    """
    Natural logarithm.  Compute log(f) where f is the decoded FP8 number.
    """
    x_float = decode(x)
    val = math.log(x_float)
    return encode(val)

def log_back(x: np.int8, y: np.int8) -> np.int8:
    """
    Backward pass for logarithm.
    Since d/dx log(x) = 1/x, multiply the upstream gradient y by 1/x.
    """
    x_float = decode(x)
    y_float = decode(y)
    grad = y_float * (1.0 / x_float) if x_float != 0 else 0.0
    return encode(grad)

def exp(x: np.int8) -> np.int8:
    """
    Exponential function: compute exp(f) where f is the decoded FP8 number.
    """
    x_float = decode(x)
    val = math.exp(x_float)
    return encode(val)

def exp_back(x: np.int8, y: np.int8) -> np.int8:
    """
    Backward pass for the exponential.
    Since d/dx exp(x) = exp(x), multiply the upstream gradient y by exp(x).
    """
    x_float = decode(x)
    y_float = decode(y)
    grad = y_float * math.exp(x_float)
    return encode(grad)

def inv(x: np.int8) -> np.int8:
    """
    Inverse function: compute 1/f.
    (Recall that in our system FP8(64) decodes to 1.)
    """
    x_float = decode(x)
    val = 1.0 / x_float
    return encode(val)

def inv_back(x: np.int8, y: np.int8) -> np.int8:
    """
    Backward pass for the inverse.
    Since d/dx (1/x) = -1/x^2, multiply upstream gradient y by -1/x^2.
    """
    x_float = decode(x)
    y_float = decode(y)
    grad = y_float * (-1.0 / (x_float * x_float)) if x_float != 0 else 0.0
    return encode(grad)

def is_close(x: np.int8, y: np.int8) -> bool:
    """Return True if the decoded values of x and y are close."""
    return math.isclose(decode(x), decode(y), rel_tol=3e-1, abs_tol=1e-1)

def lt(x: np.int8, y: np.int8) -> bool:
    """Less-than comparison using the underlying integer codes."""
    return bool(x < y)

def eq(x: np.int8, y: np.int8) -> bool:
    """Equality check using the underlying integer codes."""
    return x == y

def max_(x: np.int8, y: np.int8) -> np.int8:
    """Return the maximum (by comparing the codes)."""
    return x if x >= y else y

#-----------------------------------------------------------
# Iterable Helpers
#-----------------------------------------------------------
def map(fn: Callable[[np.int8], np.int8], l: Iterable[np.int8]) -> Iterable[np.int8]:
    """Apply fn elementwise."""
    return (fn(x) for x in l)

def zipWith(fn: Callable[[np.int8, np.int8], np.int8],
            l1: Iterable[np.int8],
            l2: Iterable[np.int8]) -> Iterable[np.int8]:
    """Apply fn to pairs from l1 and l2."""
    return (fn(x, y) for x, y in zip(l1, l2))

def reduce(fn: Callable[[np.int8, np.int8], np.int8],
           l: Iterable[np.int8],
           init: np.int8) -> np.int8:
    """Reduce the iterable l with fn starting from init."""
    acc = init
    for x in l:
        acc = fn(acc, x)
    return acc

def sum(l: Iterable[np.int8]) -> np.int8:
    """
    Sum a list of FP8 numbers using our log‑based addition.
    (Here, 0 is the FP8 encoding of ~1/256, our “zero”.)
    """
    return reduce(add, l, np.int8(0))

def prod(l: Iterable[np.int8]) -> np.int8:
    """
    Multiply a list of FP8 numbers.
    (Recall that FP8(64) encodes 1.)
    """
    return reduce(mul, l, np.int8(64))


def zeros(shape: Iterable[int]) -> np.ndarray:
    """Return a zero tensor of the given shape."""
    return np.zeros(tuple(shape), dtype=np.int8)

def rand(shape: Iterable[int]) -> np.ndarray:
    """Return a random tensor of the given shape."""
    return np.random.randint(-64, 64, tuple(shape), dtype=np.int8)

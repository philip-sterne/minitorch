import math
import numpy as np

# Import the new FP8 functions.
# (Adjust the import path as needed.)
from minitorch.ln8_precision import (
    encode,
    decode,
    add,
    neg,
    mul,
    truediv,
    relu,
    sigmoid,
    sigmoid_back,
    log,
    log_back,
    exp,
    exp_back,
    inv,
    inv_back,
    is_close,
    lt,
    eq,
    max_,
    map as fp8_map,
    zipWith,
    reduce,
    sum as fp8_sum,
    prod as fp8_prod,
)

def fp8_from_float(val):
    """
    If val is a scalar, returns a single np.int8 code.
    If val is an iterable, returns a numpy array (dtype np.int8) of codes.
    """
    if isinstance(val, (list, np.ndarray)):
        return np.array([encode(v) for v in val], dtype=np.int8)
    else:
        return encode(val)


def fp8_to_float(code):
    """
    If code is a scalar np.int8, returns the decoded float.
    If code is an array, returns a numpy array of floats.
    """
    if isinstance(code, np.ndarray):
        return np.array([decode(x) for x in code], dtype=float)
    else:
        return decode(code)


def add_array(a, b):
    """Element‐wise addition on arrays of np.int8 codes."""
    return np.array([add(x, y) for x, y in zip(a, b)], dtype=np.int8)


def sub(x, y):
    """Subtraction defined as addition with negation."""
    return add(x, neg(y))


def sub_array(a, b):
    """Element‐wise subtraction for arrays of np.int8 codes."""
    return np.array([sub(x, y) for x, y in zip(a, b)], dtype=np.int8)


def mul_array(a, b):
    """Element‐wise multiplication."""
    return np.array([mul(x, y) for x, y in zip(a, b)], dtype=np.int8)


def truediv_array(a, b):
    """Element‐wise division."""
    return np.array([truediv(x, y) for x, y in zip(a, b)], dtype=np.int8)


# For comparisons we define gt as the opposite of lt:
gt = lambda x, y: lt(y, x)
le = lambda x, y: (lt(x, y) or eq(x, y))
ge = lambda x, y: (lt(y, x) or eq(x, y))


# --- Basic arithmetic tests ----------------------------------

def test_addition_same_sign():
    a = fp8_from_float(1.0)
    b = fp8_from_float(1.0)
    c = add(a, b)
    assert math.isclose(fp8_to_float(c), 2.0, rel_tol=1e-4)


def test_addition_diff_sign():
    # 16 + (-8) => 8
    a = fp8_from_float(16.0)
    b = fp8_from_float(-8.0)
    c = add(a, b)
    assert math.isclose(fp8_to_float(c), 8.0, rel_tol=1e-2)


def test_subtraction_same_sign():
    # 4 - 2 = 2
    a = fp8_from_float(4.0)
    b = fp8_from_float(2.0)
    c = sub(a, b)
    assert math.isclose(fp8_to_float(c), 2.0, rel_tol=1e-4)


def test_subtraction_diff_sign():
    # 4 - (-4) = 8
    a = fp8_from_float(4.0)
    b = fp8_from_float(-4.0)
    c = sub(a, b)
    assert math.isclose(fp8_to_float(c), 8.0, rel_tol=1e-4)


def test_multiplication():
    # 2*(-2) = -4
    a = fp8_from_float(2.0)
    b = fp8_from_float(-2.0)
    c = mul(a, b)
    assert math.isclose(fp8_to_float(c), -4.0, rel_tol=1e-4)


def test_division():
    # 8 / 4 = 2
    a = fp8_from_float(8.0)
    b = fp8_from_float(4.0)
    c = truediv(a, b)
    assert math.isclose(fp8_to_float(c), 2.0, rel_tol=1e-4)


# --- Negation, ReLU ------------------------------------------

def test_negation_positive():
    a = fp8_from_float(2.0)
    neg_a = neg(a)
    assert math.isclose(fp8_to_float(neg_a), -2.0, rel_tol=1e-4)


def test_negation_zero_specialcase():
    # x=0 encodes ~ +1/256 => negation => ~ -1/256
    a = fp8_from_float(0.0)
    neg_a = neg(a)
    val = fp8_to_float(neg_a)
    assert math.isclose(abs(val), 1.0 / 256.0, rel_tol=1e-6)


def test_relu_negative():
    a = fp8_from_float(-3.0)
    r = relu(a)
    # ReLU: negative becomes code 0 (which decodes to ~ +1/256)
    assert math.isclose(fp8_to_float(r), 1.0 / 256.0, rel_tol=1e-6)


def test_relu_positive():
    a = fp8_from_float(3.0)
    r = relu(a)
    # For positive values, ReLU should leave the number unchanged.
    assert math.isclose(fp8_to_float(r), fp8_to_float(a), rel_tol=1e-1)


# --- Equality & Comparison ------------------------------------

def test_equality():
    a = fp8_from_float(5.0)
    b = fp8_from_float(5.0)
    c = fp8_from_float(-5.0)
    assert a == b
    assert a != c


def test_comparison():
    a = fp8_from_float(3.0)
    b = fp8_from_float(5.0)
    assert lt(a, b)
    assert not(gt(a, b))
    assert le(a, a)
    assert ge(b, a)


def test_trichotomy_basic():
    # Test a few finite values.
    values = [
        fp8_from_float(-10),
        fp8_from_float(-1),
        fp8_from_float(0),
        fp8_from_float(1),
        fp8_from_float(10),
    ]
    for i in range(len(values)):
        for j in range(len(values)):
            a = values[i]
            b = values[j]
            less = lt(a, b)
            equal = eq(a, b)
            greater = lt(b, a)  # defined as gt(a,b)
            # Exactly one of these must be True.
            assert (int(less) + int(equal) + int(greater)) == 1


# --- Extended tests ---------------------------------------------------

def test_from_float_zero():
    """Zero => code=0 => decodes to ~ +1/256."""
    x = fp8_from_float(0.0)
    assert isinstance(x, np.int8)
    assert x == 0
    assert math.isclose(fp8_to_float(x), 1 / 256, rel_tol=1e-7)


def test_from_float_pos_small():
    """Very small positive values yield code=0 and decode to ~ +1/256."""
    small_vals = [1e-50, 1e-46, 1e-45, 1e-40]
    codes = np.array([fp8_from_float(v) for v in small_vals], dtype=np.int8)
    assert np.all(codes == 0)
    for code in codes:
        assert math.isclose(fp8_to_float(code), 1 / 256, rel_tol=1e-7)


def test_from_float_big():
    """
    Large positive values saturate at +127 and large negatives at -127.
    """
    large_vals = [1e12, 1e20, 1e30, 1e40]
    codes = np.array([fp8_from_float(v) for v in large_vals], dtype=np.int8)
    assert np.all(codes == 127)
    large_negs = [-1e12, -1e20, -1e30, -1e40]
    codes_neg = np.array([fp8_from_float(v) for v in large_negs], dtype=np.int8)
    assert np.all(codes_neg == -128)


def test_from_float_typical_arrays():
    """Test a typical array of positive and negative values."""
    vals = np.array([0.5, 1.0, 2.0, -3.0, 16.0], dtype=float)
    codes = np.array([fp8_from_float(v) for v in vals], dtype=np.int8)
    assert codes.shape == (5,)
    decoded = np.array([fp8_to_float(x) for x in codes])
    assert decoded.shape == (5,)
    # Check that the negative value decodes to a negative number.
    assert decoded[3] < 0


def test_to_float_zero_code():
    """A code of 0 decodes to ~ +1/256."""
    arr = np.array([0, 0, 0], dtype=np.int8)
    decoded = np.array([fp8_to_float(x) for x in arr])
    assert np.allclose(decoded, 1 / 256)


def test_to_float_positive_codes():
    """
    For known positive codes, check the approximate decodes.
    E.g. 64 -> 2^((64-64)/8)=1, 72 -> 2, 80 -> 4.
    """
    arr = np.array([64, 72, 80], dtype=np.int8)
    decoded = np.array([fp8_to_float(x) for x in arr])
    expected = [1.0, 2.0, 4.0]
    assert np.allclose(decoded, expected, rtol=1e-6)


def test_to_float_negative_codes():
    """
    For negative codes: e.g. -64 -> -1, -72 -> -2, -80 -> -4.
    """
    arr = np.array([-64, -72, -80], dtype=np.int8)
    decoded = np.array([fp8_to_float(x) for x in arr])
    expected = [-1.0, -2.0, -4.0]
    assert np.allclose(decoded, expected, rtol=1e-6)


def test_add_same_sign():
    """Check same-sign addition (scalar and array)."""
    a = fp8_from_float(2.0)
    b = fp8_from_float(2.0)
    c = add(a, b)
    assert math.isclose(fp8_to_float(c), 4.0, rel_tol=0.1)
    # Array version:
    arrA = np.array([fp8_from_float(v) for v in [1.0, 2.0, 8.0]], dtype=np.int8)
    arrB = np.array([fp8_from_float(v) for v in [1.0, 2.0, 8.0]], dtype=np.int8)
    sum_codes = add_array(arrA, arrB)
    decoded = np.array([fp8_to_float(x) for x in sum_codes])
    assert np.allclose(decoded, [2.0, 4.0, 16.0], rtol=0.1)


def test_add_diff_sign_bigger_pos():
    """Check elementwise: (16) + (-8) => ~8, (2) + (-1) => ~1."""
    a = np.array([fp8_from_float(v) for v in [16.0, 2.0, 10.0]], dtype=np.int8)
    b = np.array([fp8_from_float(v) for v in [-8.0, -1.0, -5.0]], dtype=np.int8)
    c = add_array(a, b)
    decoded = np.array([fp8_to_float(x) for x in c])
    assert np.allclose(decoded, [8.0, 1.0, 5.0], rtol=0.15)


def test_add_diff_sign_bigger_neg():
    """(-8) + (2) => ~ -6, etc."""
    a = np.array([fp8_from_float(v) for v in [-8.0, -10.0]], dtype=np.int8)
    b = np.array([fp8_from_float(v) for v in [2.0, 1.0]], dtype=np.int8)
    c = add_array(a, b)
    decoded = np.array([fp8_to_float(x) for x in c])
    assert decoded[0] < 0
    assert decoded[1] < 0
    assert math.isclose(decoded[0], -6.0, abs_tol=1.0)
    assert math.isclose(decoded[1], -9.0, abs_tol=1.5)


def test_add_diff_sign_tie():
    """(4) + (-4) => tie: result should be code 0 and decode to ~ +1/256."""
    a = fp8_from_float(4.0)
    b = fp8_from_float(-4.0)
    c = add(a, b)
    assert c == 0
    assert math.isclose(fp8_to_float(c), 1 / 256, rel_tol=1e-7)


def test_sub_same_sign():
    """E.g., 4 - 2 => 2, 8 - 1 => 7."""
    a = np.array([fp8_from_float(v) for v in [4.0, 8.0, 2.0]], dtype=np.int8)
    b = np.array([fp8_from_float(v) for v in [2.0, 1.0, 1.0]], dtype=np.int8)
    c = sub_array(a, b)
    decoded = np.array([fp8_to_float(x) for x in c])
    assert math.isclose(decoded[0], 2.0, abs_tol=0.2)
    assert math.isclose(decoded[1], 7.0, abs_tol=1.0)
    assert math.isclose(decoded[2], 1.0, abs_tol=0.2)


def test_sub_diff_sign():
    """Check subtraction with differing signs: 4 - (-4) => 8."""
    a = fp8_from_float(4.0)
    b = fp8_from_float(-4.0)
    c = sub(a, b)
    assert fp8_to_float(c) > 7.0


def test_mul_simple():
    """Test: 2 * 2 => 4, 3 * (-2) => -6, etc."""
    a = np.array([fp8_from_float(v) for v in [2.0, 3.0, -2.0]], dtype=np.int8)
    b = np.array([fp8_from_float(v) for v in [2.0, -2.0, 4.0]], dtype=np.int8)
    c = mul_array(a, b)
    dec = np.array([fp8_to_float(x) for x in c])
    assert math.isclose(dec[0], 4.0, abs_tol=0.5)
    assert math.isclose(dec[1], -6.0, abs_tol=1.0)
    assert math.isclose(dec[2], -8.0, abs_tol=1.0)


def test_mul_clamping():
    """Multiplying huge numbers should clamp to code +127 or -127."""
    bigA = fp8_from_float(1e20)
    bigB = fp8_from_float(1e20)
    c = mul(bigA, bigB)
    assert c == 127
    assert math.isclose(fp8_to_float(c), 239, abs_tol=50)
    negC = mul(bigA, fp8_from_float(-1e20))
    assert negC == -127


def test_div_basic():
    """8/4 => 2, 12/(-3) => -4, etc."""
    a = np.array([fp8_from_float(v) for v in [8.0, 12.0]], dtype=np.int8)
    b = np.array([fp8_from_float(v) for v in [4.0, -3.0]], dtype=np.int8)
    c = truediv_array(a, b)
    dec = np.array([fp8_to_float(x) for x in c])
    assert math.isclose(dec[0], 2.0, abs_tol=0.5)
    assert math.isclose(dec[1], -4.0, abs_tol=1.0)


# --- Negation and ReLU (array versions) --------------------------------

def test_negation_zero():
    """0 encodes to 0; negation should return 0."""
    zero = fp8_from_float(0.0)
    negz = neg(zero)
    assert negz == 0


def test_negation_general():
    """Negation: positive becomes negative and vice versa."""
    codes = np.array([fp8_from_float(v) for v in [2.0, -3.0, 0.0]], dtype=np.int8)
    neg_codes = np.array([neg(x) for x in codes], dtype=np.int8)
    dec = np.array([fp8_to_float(x) for x in neg_codes])
    assert dec[0] < 0
    assert dec[1] > 0
    assert math.isclose(dec[2], 1 / 256, rel_tol=1e-7)


def test_relu_negative_array():
    arr = np.array([fp8_from_float(v) for v in [-5.0, -1.0, 2.0, 10.0]], dtype=np.int8)
    r = np.array([relu(x) for x in arr], dtype=np.int8)
    dec = np.array([fp8_to_float(x) for x in r])
    assert math.isclose(dec[0], 1 / 256, rel_tol=1e-7)
    assert math.isclose(dec[1], 1 / 256, rel_tol=1e-7)
    assert dec[2] > 1.0
    assert dec[3] > 5.0


# --- Comparisons ----------------------------------------------------------

def test_eq_scalar():
    a = np.array([fp8_from_float(v) for v in [2.0, 3.0]], dtype=np.int8)
    b = np.array([fp8_from_float(v) for v in [2.0, 3.001]], dtype=np.int8)
    assert eq(a[0], b[0])
    assert eq(a[1], b[1])


def test_lt_gt_mixed():
    a = np.array([fp8_from_float(v) for v in [1.0, -5.0, 3.0]], dtype=np.int8)
    b = np.array([fp8_from_float(v) for v in [2.0, -10.0, 3.0]], dtype=np.int8)
    assert lt(a[0], b[0])
    assert not(lt(a[1], b[1]))
    assert not(lt(a[2], b[2]))
    assert not(gt(a[0], b[0]))
    assert gt(a[1], b[1])
    assert not(gt(a[2], b[2]))
    assert not eq(a[0], b[0])
    assert not eq(a[1], b[1])
    assert eq(a[2], b[2])


# --- Exhaustive tests -----------------------------------------------------

def test_encode():
    """Check that encoding and decoding are consistent."""
    for i in range(-128, 128):
        sign = math.copysign(1, i)
        print(f"i={i}, sign={sign}")
        val = sign * 2 ** ((abs(i)-64) / 8)
        print(f"val={val}")
        print(f"encode(val)={encode(val)}")
        assert encode(val) == i

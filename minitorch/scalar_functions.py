from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import precision
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike

ops = precision.CURRENT_PRECISION
dtype = precision.CURRENT_TYPE


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: dtype) -> Tuple[dtype, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: dtype) -> dtype:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, dtype), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: dtype, b: dtype) -> dtype:
        return dtype(a + b)

    @staticmethod
    def backward(ctx: Context, d_output: dtype) -> Tuple[dtype, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: dtype) -> dtype:
        ctx.save_for_backward(a)
        return ops.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: dtype) -> dtype:
        (a,) = ctx.saved_values
        return ops.log_back(a, d_output)


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: dtype, b: dtype) -> dtype:
        ctx.save_for_backward(a, b)
        return ops.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: dtype) -> Tuple[dtype, dtype]:
        (a, b) = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: dtype) -> dtype:
        ctx.save_for_backward(a)
        return ops.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: dtype) -> dtype:
        (a,) = ctx.saved_values
        return ops.inv_back(a, d_output)


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: dtype) -> dtype:
        return ops.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: dtype) -> dtype:
        return ops.neg(d_output)


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: dtype) -> dtype:
        ctx.save_for_backward(a)
        return ops.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: dtype) -> dtype:
        (a,) = ctx.saved_values
        return ops.sigmoid_back(a, d_output)


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: dtype) -> dtype:
        ctx.save_for_backward(a)

        return ops.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: dtype) -> dtype:
        (a,) = ctx.saved_values
        return ops.relu_back(a, d_output)


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: dtype) -> dtype:
        ctx.save_for_backward(a)
        return ops.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: dtype) -> dtype:
        (a,) = ctx.saved_values
        return ops.exp_back(a, d_output)


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: dtype, b: dtype) -> dtype:
        return dtype(ops.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: dtype) -> Tuple[dtype, dtype]:
        return (dtype(0), dtype(0))


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: dtype, b: dtype) -> dtype:
        return dtype(ops.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: dtype) -> Tuple[dtype, dtype]:
        return dtype(0), dtype(0)

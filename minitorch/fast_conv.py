from typing import Tuple, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    OutIndex,
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # Determine the offset. If reverse is True, the filter is anchored on the
    # right so that its last element (index kw-1) aligns with the current output.
    # Otherwise, it is anchored on the left.
    offset = kw - 1 if reverse else 0
    index = np.zeros(3, dtype=np.int32)

    # Iterate over each output element in parallel.
    for i in prange(out_size):
        # Get the multi-index (b, oc, x) corresponding to the flat index i.
        to_index(i, out_shape, index)
        b, oc, x = index
        tmp = 0.0
        # Sum over all input channels and kernel positions.
        for ic in range(in_channels):
            for k in range(kw):
                # Compute the corresponding input index.
                ix = x + k - offset
                # If the index is out of bounds, we treat the input as zero (padded).
                if ix < 0 or ix >= width:
                    continue
                # Compute the flat positions for the input and weight.
                pos_in = (
                    b * input_strides[0] + ic * input_strides[1] + ix * input_strides[2]
                )
                pos_w = (
                    oc * weight_strides[0]
                    + ic * weight_strides[1]
                    + k * weight_strides[2]
                )
                tmp += input[pos_in] * weight[pos_w]
        # Write the computed sum to the output.
        pos_out = b * out_strides[0] + oc * out_strides[1] + x * out_strides[2]
        out[pos_out] = tmp


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of shape
       `batch, in_channels, height, width`

    and weight tensor of shape
       `out_channels, in_channels, k_height, k_width`

    Computes padded output of shape
       `batch, out_channels, height, width`

    `reverse` decides if the weight is anchored at the top-left (False)
    or bottom-right (True). For example, if `reverse` is True then the kernel
    element at index (kh-1, kw-1) will align with the current output pixel.

    """
    # Unpack shapes
    batch_out, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_w, in_channels_w, kh, kw = weight_shape

    # Ensure the input dimensions are consistent.
    assert (
        batch == batch_out
        and in_channels == in_channels_w
        and out_channels == out_channels_w
    )

    # Determine offsets: if reverse is True, anchor kernel's bottom/right edge with the output.
    offset_y = kh - 1 if reverse else 0
    offset_x = kw - 1 if reverse else 0

    # Create a temporary index array for converting flat indices into multi-indices.
    index = np.zeros(4, dtype=np.int32)

    # Iterate over every element in the output tensor.
    for i in prange(out_size):
        # Convert flat index i into multi-index (b, oc, y, x) in the output tensor.
        to_index(i, out_shape, index)
        b, oc, y, x = index[0], index[1], index[2], index[3]

        tmp = 0.0
        # Iterate over each input channel.
        for ic in range(in_channels):
            # Iterate over the kernel height.
            for i_k in range(kh):
                # Iterate over the kernel width.
                for j_k in range(kw):
                    # Compute corresponding input spatial indices.
                    in_y = y + i_k - offset_y
                    in_x = x + j_k - offset_x
                    # If the input index is out of bounds, treat it as zero (i.e. padded).
                    if in_y < 0 or in_y >= height or in_x < 0 or in_x >= width:
                        continue
                    # Compute flat positions using the provided strides.
                    pos_in = (
                        b * input_strides[0]
                        + ic * input_strides[1]
                        + in_y * input_strides[2]
                        + in_x * input_strides[3]
                    )
                    pos_w = (
                        oc * weight_strides[0]
                        + ic * weight_strides[1]
                        + i_k * weight_strides[2]
                        + j_k * weight_strides[3]
                    )
                    tmp += input[pos_in] * weight[pos_w]
        # Compute the flat position for the output element and store the result.
        pos_out = (
            b * out_strides[0]
            + oc * out_strides[1]
            + y * out_strides[2]
            + x * out_strides[3]
        )
        out[pos_out] = tmp


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply

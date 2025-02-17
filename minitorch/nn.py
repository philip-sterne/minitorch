from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor, Max


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # First reshape to (B, C, new_height, kh, new_width, kw)
    t = input.view(batch, channel, new_height, kh, new_width, kw).contiguous()
    # Permute to (B, C, new_height, new_width, kh, kw)
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    # Flatten the kernel dimensions: (B, C, new_height, new_width, kh*kw)
    t = t.view(batch, channel, new_height, new_width, kh * kw)
    return t, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
        input: Tensor of shape (batch, channel, height, width).
        kernel: Tuple (kh, kw) for the pooling kernel dimensions.

    Returns:
        A tensor of shape (batch, channel, new_height, new_width) where
        each output element is the average of a kernel window.
    """
    tiled, new_height, new_width = tile(input, kernel)
    kh, kw = kernel
    pool_area = kh * kw
    # Sum over the kernel window (dimension 4) then squeeze that dimension.
    return tiled.sum(dim=4) / pool_area


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax of the input tensor along a given dimension.

    Args:
        input: The input tensor.
        dim: The dimension to compute the argmax along.
    """
    max_indices = input.zeros(input.shape).long()
    max_vals = input.zeros(input.shape)

    # Initialize with first element
    for i in range(input.shape[dim]):
        # Get slice along dimension
        slc = input.index_select(dim, i)
        # Update max values and indices where new values are larger
        max_indices = max_indices.where(slc <= max_vals, i)
        max_vals = max_vals.maximum(slc)

    return max_indices


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max of the input tensor along a given dimension.

    Args:
        input: The input tensor.
        dim: The dimension to compute the max along.
    """
    return Max.apply(input, dim)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
        input: Tensor of shape (batch, channel, height, width).
        kernel: Tuple (kh, kw) for the pooling kernel dimensions.
    """
    tiled, new_height, new_width = tile(input, kernel)
    return Max.apply(tiled, dim=4)


def softmax(input: Tensor) -> Tensor:
    """Compute the softmax of the input tensor."""
    # Compute softmax along the last dimension in a numerically stable way.
    last_dim = len(input.shape) - 1
    # Subtract the maximum value in the last dimension (with unsqueeze for broadcasting)
    max_vals = max(input, dim=last_dim).unsqueeze(last_dim)
    # Exponentiate the stabilized tensor
    exp_vals = (input - max_vals).exp()
    # Sum the exponentials over the last dimension (with unsqueeze for proper broadcasting)
    sum_exp = exp_vals.sum(dim=last_dim).unsqueeze(last_dim)
    # Return the normalized probabilities
    return exp_vals / sum_exp


def logsoftmax(input: Tensor) -> Tensor:
    """Compute the log of the softmax of the input tensor."""
    # Compute softmax along the last dimension in a numerically stable way.
    last_dim = len(input.shape) - 1
    # Subtract the maximum value in the last dimension (with unsqueeze for broadcasting)
    max_vals = max(input, dim=last_dim).unsqueeze(last_dim)
    # Exponentiate the stabilized tensor
    exp_vals = (input - max_vals).exp()
    # Sum the exponentials over the last dimension (with unsqueeze for proper broadcasting)
    sum_exp = exp_vals.sum(dim=last_dim).unsqueeze(last_dim)
    # Return the normalized probabilities
    return max_vals - sum_exp.log()


def dropout(
    input: Tensor, p: float, train: bool = True, ignore: bool = False
) -> Tensor:
    """Apply dropout to the input tensor.

    Args:
        input: The input tensor.
        p: Dropout probability (fraction of elements to drop).
        train: If False, dropout is turned off.
        ignore: If True, dropout is ignored.

    Returns:
        The tensor after applying dropout.
    """
    if ignore or (not train) or p == 0:
        return input
    if p == 1.0:
        return input.zeros(input.shape)
    noise = rand(input.shape, backend=input.backend)
    mask = noise > p
    return input * mask / (1.0 - p)

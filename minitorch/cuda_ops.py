# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps
import minitorch.precision

dtype = minitorch.precision.CURRENT_TYPE

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[dtype], dtype]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[dtype], dtype] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[dtype, dtype], dtype]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[dtype, dtype], dtype] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[dtype, dtype], dtype], start: dtype = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[dtype, dtype], dtype] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[dtype], dtype],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i >= out_size:
            return  # out-of-range thread

        # Determine the number of dimensions in the output and input.
        # (We assume that out_shape and in_shape are device arrays of int32.)
        ndim_out = out_shape.shape[0]
        ndim_in = in_shape.shape[0]

        # --- 1. Convert flat index i into multi-index for the output tensor ---
        temp = i
        # Process dimensions from last to first.
        for d in range(ndim_out - 1, -1, -1):
            # Use modulus to get the index for dimension d.
            out_index[d] = temp % out_shape[d]
            temp = temp // out_shape[d]

        # --- 2. Compute the flat output position using strides ---
        out_ordinal = 0
        for d in range(ndim_out):
            out_ordinal += out_index[d] * out_strides[d]

        # --- 3. Compute the broadcast input index ---
        # For broadcasting, we assume that the input tensor’s dimensions align
        # to the right of the output tensor’s dimensions.
        offset = ndim_out - ndim_in
        for d in range(ndim_in):
            # If the input dimension is 1, then that axis is broadcast:
            if in_shape[d] == 1:
                in_index[d] = 0
            else:
                in_index[d] = out_index[d + offset]

        # --- 4. Compute the flat input position using the input strides ---
        in_ordinal = 0
        for d in range(ndim_in):
            in_ordinal += in_index[d] * in_strides[d]

        # --- 5. Apply the mapping function and write the result ---
        out[out_ordinal] = fn(in_storage[in_ordinal])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[dtype, dtype], dtype],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i >= out_size:
            return  # Exit if the thread index is out of range.

        # Determine the number of dimensions.
        ndim_out = out_shape.shape[0]
        ndim_a = a_shape.shape[0]
        ndim_b = b_shape.shape[0]

        # --- 1. Convert flat index i into a multi-index for the output tensor ---
        temp = i
        # Compute the multi-index by processing dimensions from last to first.
        for d in range(ndim_out - 1, -1, -1):
            out_index[d] = temp % out_shape[d]
            temp = temp // out_shape[d]

        # --- 2. Compute the flat output ordinal using output strides ---
        out_ordinal = 0
        for d in range(ndim_out):
            out_ordinal += out_index[d] * out_strides[d]

        # --- 3. Compute the broadcasted multi-index for the first input tensor ---
        # Align dimensions to the right.
        offset_a = ndim_out - ndim_a
        for d in range(ndim_a):
            # If this dimension is broadcast (size 1), index is always 0.
            if a_shape[d] == 1:
                a_index[d] = 0
            else:
                a_index[d] = out_index[d + offset_a]

        # --- 4. Compute the broadcasted multi-index for the second input tensor ---
        offset_b = ndim_out - ndim_b
        for d in range(ndim_b):
            if b_shape[d] == 1:
                b_index[d] = 0
            else:
                b_index[d] = out_index[d + offset_b]

        # --- 5. Compute flat positions in the input storages using strides ---
        a_ordinal = 0
        for d in range(ndim_a):
            a_ordinal += a_index[d] * a_strides[d]

        b_ordinal = 0
        for d in range(ndim_b):
            b_ordinal += b_index[d] * b_strides[d]

        # --- 6. Apply the binary function and store the result in the output ---
        out[out_ordinal] = fn(a_storage[a_ordinal], b_storage[b_ordinal])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Load data from global memory to shared memory.
    # If i is out-of-bounds, load 0 to avoid affecting the sum.
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0
    cuda.syncthreads()

    # Perform reduction in shared memory.
    # Each step halves the number of active threads.
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()  # Wait for all threads to update their shared memory.
        stride //= 2

    # The first thread in the block writes the result to the output.
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[dtype, dtype], dtype],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: dtype,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        a_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Make sure the block index is valid.
        if out_pos >= out_size:
            return

        # Determine the number of dimensions (assumed stored in out_shape).
        ndim = out_shape.shape[0]

        # --- 1. Convert block id (flat index) to multi-index for the output tensor ---
        temp = out_pos
        for d in range(ndim - 1, -1, -1):
            out_index[d] = temp % out_shape[d]
            temp = temp // out_shape[d]

        # --- 2. Establish the base multi-index for the input tensor ---
        # For every dimension other than reduce_dim, the input index is the same as the output.
        for d in range(ndim):
            a_index[d] = out_index[d]
        # (The value for dimension reduce_dim will be supplied by the loop below.)

        # --- 3. Each thread performs a partial reduction over the reduction axis ---
        partial = reduce_value  # initialize with the identity value
        # The total number of elements along the reduction dimension.
        r_limit = a_shape[reduce_dim]
        r = pos
        while r < r_limit:
            # Set the coordinate for the reduction axis.
            a_index[reduce_dim] = r
            # Compute the flat index into a_storage for the current multi-index.
            a_ord = 0
            for d in range(ndim):
                a_ord += a_index[d] * a_strides[d]
            # Accumulate into partial using the reduction function.
            partial = fn(partial, a_storage[a_ord])
            r += cuda.blockDim.x  # move to the next element for this thread

        # Write this thread's partial result to shared memory.
        cache[pos] = partial
        cuda.syncthreads()

        # --- 4. Reduce the partial results in shared memory ---
        stride = cuda.blockDim.x // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2

        # --- 5. Write the final reduced value from this block to the output ---
        if pos == 0:
            # Compute the flat index into the output using out_index and out_strides.
            out_ord = 0
            for d in range(ndim):
                out_ord += out_index[d] * out_strides[d]
            out[out_ord] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    sA = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    sB = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Get 2D thread indices.
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Since size may be less than BLOCK_DIM, only threads with indices < size load data.
    if ty < size and tx < size:
        # Each element of a is located at a[ty * size + tx]
        sA[ty, tx] = a[ty * size + tx]
        sB[ty, tx] = b[ty * size + tx]
    # Ensure all threads have loaded their data into shared memory.
    cuda.syncthreads()

    # Each thread (with indices within the [size, size] region) computes one element of out.
    tmp = 0.0
    if ty < size and tx < size:
        for k in range(size):
            tmp += sA[ty, k] * sB[k, tx]

    # (Optional: a sync here is not strictly required because each thread uses only its own computed value.)
    # Write the computed result back to global memory.
    if ty < size and tx < size:
        out[ty * size + tx] = tmp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    # Use a tile size (BLOCK_DIM) of 32.
    BLOCK_DIM = 32

    # Allocate shared memory tiles for A and B.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Get thread indices within the block.
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Get global indices for the output element.
    # Here, we assume that the output is 3D: (batch, m, n)
    # with m = number of rows of A and n = number of columns of B.
    batch = cuda.blockIdx.z
    i = cuda.blockIdx.x * cuda.blockDim.x + tx  # row index in A
    j = cuda.blockIdx.y * cuda.blockDim.y + ty  # col index in B

    # Determine matrix dimensions.
    # a_shape is assumed to be (batch, m, K) and b_shape (batch, K, n).
    m = a_shape[a_shape.shape[0] - 2]
    K = a_shape[a_shape.shape[0] - 1]
    n = b_shape[b_shape.shape[0] - 1]

    # Compute batch offsets using the first dimension.
    a_batch_offset = 0
    if a_shape[0] > 1:
        a_batch_offset = batch * a_strides[0]
    b_batch_offset = 0
    if b_shape[0] > 1:
        b_batch_offset = batch * b_strides[0]
    out_batch_offset = 0
    if out_shape[0] > 1:
        out_batch_offset = batch * out_strides[0]

    # Initialize accumulator for the dot-product.
    partial = 0.0

    # Loop over the tiles along the inner (K) dimension.
    for t in range(0, K, BLOCK_DIM):
        # Load one tile of A into shared memory.
        # A is of shape (m, K) for this batch.
        if i < m and (t + tx) < K:
            # Compute flat index into a_storage:
            # a[i, t+tx] => a_batch_offset + i * a_strides[1] + (t+tx)*a_strides[2]
            a_shared[ty, tx] = a_storage[
                a_batch_offset + i * a_strides[1] + (t + tx) * a_strides[2]
            ]
        else:
            a_shared[ty, tx] = 0.0

        # Load one tile of B into shared memory.
        # B is of shape (K, n) for this batch.
        if (t + ty) < K and j < n:
            # b[t+ty, j] => b_batch_offset + (t+ty)*b_strides[1] + j*b_strides[2]
            b_shared[ty, tx] = b_storage[
                b_batch_offset + (t + ty) * b_strides[1] + j * b_strides[2]
            ]
        else:
            b_shared[ty, tx] = 0.0

        # Ensure the entire tile is loaded.
        cuda.syncthreads()

        # Multiply the two tiles together.
        for k in range(BLOCK_DIM):
            partial += a_shared[ty, k] * b_shared[k, tx]

        # Wait for all threads before loading the next tile.
        cuda.syncthreads()

    # Write the computed value to the output if within bounds.
    if i < m and j < n:
        # Compute the flat index into out_storage:
        # out[batch, i, j] = out_batch_offset + i*out_strides[1] + j*out_strides[2]
        index = out_batch_offset + i * out_strides[1] + j * out_strides[2]
        out[index] = partial


tensor_matrix_multiply = jit(_tensor_matrix_multiply)

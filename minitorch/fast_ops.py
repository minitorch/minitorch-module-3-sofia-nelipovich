from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
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

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

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
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        size = 1
        for s in out_shape:
            size *= s
        # Проверка shape/strides без np.array_equal
        same_shape = True
        if len(out_shape) != len(in_shape):
            same_shape = False
        else:
            for i in range(len(out_shape)):
                if out_shape[i] != in_shape[i]:
                    same_shape = False
                    break
        same_strides = True
        if len(out_strides) != len(in_strides):
            same_strides = False
        else:
            for i in range(len(out_strides)):
                if out_strides[i] != in_strides[i]:
                    same_strides = False
                    break
        if same_shape and same_strides:
            for i in prange(size):
                out[i] = fn(in_storage[i])
        else:
            for ordinal in prange(size):
                # out_index = [0]*len(out_shape)
                # in_index = [0]*len(in_shape)
                out_index = np.empty(len(out_shape), np.int64)
                in_index = np.empty(len(in_shape), np.int64)
                curr = ordinal
                # to_index
                for d in range(len(out_shape)-1, -1, -1):
                    out_index[d] = curr % out_shape[d]
                    curr //= out_shape[d]
                # broadcast_index
                in_offset = len(out_shape) - len(in_shape)
                for i in range(len(in_shape)):
                    if in_shape[i] == 1:
                        in_index[i] = 0
                    else:
                        in_index[i] = out_index[i + in_offset]
                # index_to_position
                out_pos = 0
                for d in range(len(out_shape)):
                    out_pos += out_index[d] * out_strides[d]
                in_pos = 0
                for d in range(len(in_shape)):
                    in_pos += in_index[d] * in_strides[d]
                out[out_pos] = fn(in_storage[in_pos])
       
    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """
    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        size = 1
        for s in out_shape:
            size *= s
        # если форма и strides совпадают — прямой проход по памяти  
        if (np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
            and np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)):
            for i in prange(size):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # Общее решение через индексы
            for ordinal in prange(size):
                out_index = np.zeros(len(out_shape), np.int32)
                a_index = np.zeros(len(a_shape), np.int32)
                b_index = np.zeros(len(b_shape), np.int32)
                # Раскладываем ordinal в многомерный out_index
                # to_index
                curr = ordinal
                for d in range(len(out_shape)-1, -1, -1):
                    out_index[d] = curr % out_shape[d]
                    curr //= out_shape[d]
                # Вычисляем a_index по правилам broadcasting
                # broadcast_index
                a_offset = len(out_shape) - len(a_shape)
                for i in range(len(a_shape)):
                    if a_shape[i] == 1:
                        a_index[i] = 0
                    else:
                        a_index[i] = out_index[i + a_offset]
                # broadcast_index
                b_offset = len(out_shape) - len(b_shape)
                for i in range(len(b_shape)):
                    if b_shape[i] == 1:
                        b_index[i] = 0
                    else:
                        b_index[i] = out_index[i + b_offset]
                # index_to_position
                out_pos = 0
                for d in range(len(out_shape)):
                    out_pos += out_index[d] * out_strides[d]
                a_pos = 0
                for d in range(len(a_shape)):
                    a_pos += a_index[d] * a_strides[d]
                b_pos = 0
                for d in range(len(b_shape)):
                    b_pos += b_index[d] * b_strides[d]

                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        size = 1
        for s in out_shape:
            size *= s
        # Для прохода по осям
        reduce_dim_size = a_shape[reduce_dim]

        for ordinal in prange(size):
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            a_index = np.zeros(len(a_shape), dtype=np.int32)
            # Переводим ordinal -> многомерный out_index
            curr = ordinal
            for d in range(len(out_shape)-1, -1, -1):
                out_index[d] = curr % out_shape[d]
                curr //= out_shape[d]
            # Формируем базовый a_index (копируем, кроме reduce_dim = 0)
            a_index[:] = 0
            for i in range(len(a_shape)):
                if i < len(out_shape):
                    a_index[i] = out_index[i]

            a_pos = 0
            for d in range(len(a_shape)):
                a_pos += a_index[d] * a_strides[d]
            acc = a_storage[a_pos]
            # Внутренняя редукция по reduce_dim: 
            for r in range(1, reduce_dim_size):
                a_index[reduce_dim] = r
                # index_to_position
                a_pos = 0
                for d in range(len(a_shape)):
                    a_pos += a_index[d] * a_strides[d]
                value = a_storage[a_pos]
                acc = fn(acc, value)
            # Позиция в аутпуте
            out_pos = 0
            for d in range(len(out_shape)):
                out_pos += out_index[d] * out_strides[d]
            out[out_pos] = acc

    return njit(_reduce, parallel=True)  # type: ignore

def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Example shapes:
    # a: (batch, M, K)
    # b: (batch, K, N)
    # out: (batch, M, N)

    batch = out_shape[0] if len(out_shape) == 3 else 1
    M = out_shape[-2]
    N = out_shape[-1]
    K = a_shape[-1]  # == b_shape[-2]
    assert K == b_shape[-2]
    # Внешний цикл по всем (batch, M, N)
    for b in prange(batch):
        for i in range(M):
            for j in range(N):
                acc = 0.0
                for k in range(K):
                    # Вычисление позиции в хранилище для каждого тензора
                    # Для broadcast batch
                    a_batch = b 
                    b_batch = b
                    a_pos = (
                        a_batch * a_strides[0]
                        + i * a_strides[-2]
                        + k * a_strides[-1]
                    )
                    b_pos = (
                        b_batch * b_strides[0]
                        + k * b_strides[-2]
                        + j * b_strides[-1]
                    )
                    acc += a_storage[a_pos] * b_storage[b_pos]
                out_batch = b
                out_pos = (
                    out_batch * out_strides[0]
                    + i * out_strides[-2]
                    + j * out_strides[-1]
                )
                out[out_pos] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None

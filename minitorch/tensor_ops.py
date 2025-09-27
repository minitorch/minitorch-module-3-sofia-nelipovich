from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
            new tensor data
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
            :class:`TensorData` : new tensor data
        """

        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """
        Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
            :class:`TensorData` : new tensor
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(fn: Callable[[float], float]) -> Any:
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # tensor_map применяет функцию fn ко всем элементам входного тензора и записывает результат в out_storage
        # Пример (broadcast-версия):
        # Вход: in_storage = [10, 20], in_shape = (2,), in_strides = (1,)
        # Выход: out_storage = [0, 0, 0, 0, 0, 0], out_shape = (3, 2), out_strides = (2, 1)
        # Функция: fn = lambda x: x * 2
        #
        # Алгоритм:
        # Для out_index = (0, 0): broadcast_index → in_index = (0,), in_storage[0]=10 → out[0]=20
        # Для out_index = (0, 1): broadcast_index → in_index = (1,), in_storage[1]=20 → out[1]=40
        # Для out_index = (1, 0): broadcast_index → in_index = (0,), in_storage[0]=10 → out[2]=20
        # Для out_index = (1, 1): broadcast_index → in_index = (1,), in_storage[1]=20 → out[3]=40
        # Для out_index = (2, 0): broadcast_index → in_index = (0,), in_storage[0]=10 → out[4]=20
        # Для out_index = (2, 1): broadcast_index → in_index = (1,), in_storage[1]=20 → out[5]=40
        #
        # Итог:
        # out_storage = [20, 40, 20, 40, 20, 40]
        #
        # То есть input автоматически расширился до shape (3, 2), а функция fn применилась по каждому элементу.

        # Вычислим общий размер выходного тензора
        size = 1
        for s in out_shape:
            size *= s
        out_index = np.zeros_like(out_shape)
        in_index = np.zeros_like(in_shape)
        for ordinal in range(size):
            # Получаем многомерный индекс out_index
            to_index(ordinal, out_shape, out_index)
            # Приводим к индексу входного тензора через broadcasting
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # Получаем позиции в памяти для out и in
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            # Применяем функцию и записываем результат
            out[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
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
        # tensor_zip применяет функцию fn ко всем парам элементов из a_storage и b_storage, согласованных по broadcasting
        # Пример (broadcast-версия):
        # a_storage = [1, 2], a_shape = (2,), a_strides = (1,)
        # b_storage = [10, 20, 30, 40, 50, 60], b_shape = (3, 2), b_strides = (2, 1)
        # out_storage = [0, 0, 0, 0, 0, 0], out_shape = (3, 2), out_strides = (2, 1)
        # Функция: fn = lambda a, b: a + b
        # Для out_index = (0, 0):
        #     broadcast_index(out_index, out_shape, a_shape) → a_index = (0,)
        #     broadcast_index(out_index, out_shape, b_shape) → b_index = (0, 0)
        #     a_storage[0]=1, b_storage[0]=10 → out[0] = 11
        # Для (0,1): a_index=(1,), b_index=(0,1), a_storage[1]=2, b_storage[1]=20 → out[1]=22
        # Для (1,0): a_index=(0,), b_index=(1,0), a_storage[0]=1, b_storage[2]=30 → out[2]=31
        # Для (1,1): a_index=(1,), b_index=(1,1), a_storage[1]=2, b_storage[3]=40 → out[3]=42
        # Для (2,0): a_index=(0,), b_index=(2,0), a_storage[0]=1, b_storage[4]=50 → out[4]=51
        # Для (2,1): a_index=(1,), b_index=(2,1), a_storage[1]=2, b_storage[5]=60 → out[5]=62
        # Итог: out_storage = [11, 22, 31, 42, 51, 62]

        # Вычислим общий размер выходного тензора
        size = 1
        for s in out_shape:
            size *= s
        # Массивы для хранения многомерных индексов
        out_index = np.zeros_like(out_shape)
        a_index = np.zeros_like(a_shape)
        b_index = np.zeros_like(b_shape)
        for ordinal in range(size):
            # Получаем многомерный индекс в выходном массиве
            to_index(ordinal, out_shape, out_index)
            # Получаем согласованный индекс для первого входа
            broadcast_index(out_index, out_shape, a_shape, a_index)
            # Получаем согласованный индекс для второго входа
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # Преобразуем индексы в позиции в памяти с учётом strides
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            # Применяем функцию к паре элементов и сохраняем результат
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])
    return _zip


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
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
        # Пусть есть 2-мерный тензор размера (3, 2):
        # a_storage = [1, 2, 3, 4, 5, 6]
        # Его структура:
        # Строка 0: 1 2
        # Строка 1: 3 4
        # Строка 2: 5 6
        #
        # Редуцируем по размерности 0 (строки), то есть хотим получить сумму по каждому столбцу.
        # Форма выходного тензора out_shape = (1, 2)
        # Выходной storage на старте: out_storage = [0, 0]
        #
        # Алгоритм работы:
        # Для первого столбца:
        # - Берём элементы (0,0), (1,0), (2,0): это 1, 3, 5.
        # - Суммируем: 1 + 3 + 5 = 9 → out_storage[0] = 9
        #
        # Для второго столбца:
        # - Берём элементы (0,1), (1,1), (2,1): это 2, 4, 6.
        # - Суммируем: 2 + 4 + 6 = 12 → out_storage[1] = 12
        #
        # Итог:
        # out_storage = [9, 12]
        size = 1
        for s in out_shape:
            size *= s
        out_index = np.zeros_like(out_shape)
        a_index = np.zeros_like(a_shape)
        for ordinal in range(size):
            # Получаем многомерный индекс на выходе
            to_index(ordinal, out_shape, out_index)
            # Восстанавливаем индекс для a
            # Копируем out_index в a_index, кроме reduce_dim
            for i in range(len(a_shape)):
                if i == reduce_dim:
                    a_index[i] = 0
                else:
                    a_index[i] = out_index[i]
            acc = a_storage[index_to_position(a_index, a_strides)]
            for r in range(1, a_shape[reduce_dim]):
                a_index[reduce_dim] = r
                acc = fn(acc, a_storage[index_to_position(a_index, a_strides)])
            out[index_to_position(out_index, out_strides)] = acc

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)

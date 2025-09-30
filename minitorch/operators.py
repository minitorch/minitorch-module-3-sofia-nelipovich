"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List, Any

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# Task 0.1

def mul(x: float, y: float) -> float:
    """
    Multiply two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: Product of x and y.
    """
    return x * y


def id(x: float) -> float:
    """
    Return the identity of the given number.

    Args:
        x (float): Input number.

    Returns:
        float: x itself.
    """
    return x


def add(x: float, y: float) -> float:
    """
    Add two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: Sum of x and y.
    """
    return x + y


def neg(x: float) -> float:
    """
    Negate a number.

    Args:
        x (float): Input number.

    Returns:
        float: The negated value of x.
    """
    return -1.0 * x


def lt(x: float, y: float) -> bool:
    """
    Check if the first number is less than the second.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        bool: True if x < y, otherwise False.
    """
    return x < y


def eq(x: float, y: float) -> bool:
    """
    Check if two numbers are equal.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        bool: True if x equals y, otherwise False.
    """
    return x == y


def max(x: float, y: float) -> float:
    """
    Return the maximum of two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The greater number among x and y.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """
    Check if two numbers are approximately equal within 1e-2.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        bool: True if abs(x - y) < 1e-2, otherwise False.
    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """
    Apply the sigmoid function to the input.

    Args:
        x (float): Input number.

    Returns:
        float: The sigmoid of x.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def relu(x: float) -> float:
    """
    Apply the ReLU (rectified linear unit) function.

    Args:
        x (float): Input number.

    Returns:
        float: x if x > 0, else 0.
    """
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """
    Return the natural logarithm of x.

    Args:
        x (float): Input number.

    Returns:
        float: Natural logarithm of x.
    """
    return math.log(x)


def exp(x: float) -> float:
    """
    Return the exponential of x.

    Args:
        x (float): Input number.

    Returns:
        float: Exponential (e^x) of x.
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """
    Compute the backward pass for the log operation (chain rule).

    Args:
        x (float): Input value.
        d (float): Upstream gradient.

    Returns:
        float: The result of backpropagating through log.
    """
    return d / x


def inv(x: float) -> float:
    """
    Compute the multiplicative inverse of x.

    Args:
        x (float): Input number.

    Returns:
        float: 1 divided by x.
    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """
    Compute the backward pass for the inverse operation.

    Args:
        x (float): Input value.
        d (float): Upstream gradient.

    Returns:
        float: The result of backpropagating through the inverse.
    """
    return -1.0 * d / (x * x)


def relu_back(x: float, d: float) -> float:
    """
    Compute the backward pass for the ReLU operation.

    Args:
        x (float): Input value.
        d (float): Upstream gradient.

    Returns:
        float: d if x > 0, else 0.
    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

# Task 0.3


def map_fn(fn: Callable[[Any], Any], xs: Iterable[Any]) -> List[Any]:
    """
    Apply a function to each element in the iterable.

    Args:
        fn (Callable): Function to apply.
        xs (Iterable): Iterable of input elements.

    Returns:
        List: List of results after applying fn.
    """
    return [fn(x) for x in xs]


def zipWith(fn: Callable[[Any, Any], Any], xs: Iterable[Any], ys: Iterable[Any]) -> List[Any]:
    """
    Apply a binary function to corresponding pairs from two iterables.

    Args:
        fn (Callable): Function taking two arguments.
        xs (Iterable): First iterable.
        ys (Iterable): Second iterable.

    Returns:
        List: List of function results for each pair.
    """
    return [fn(x, y) for x, y in zip(xs, ys)]


def reduce_fn(fn: Callable[[Any, Any], Any], xs: Iterable[Any], start: Any) -> Any:
    """
    Reduce an iterable to a single value by repeatedly applying a binary function.

    Args:
        fn (Callable): Function taking two arguments, used to combine results.
        xs (Iterable): Iterable of input elements.
        start (Any): Initial value for the accumulator.

    Returns:
        Any: The reduced value.
    """
    res = start
    for x in xs:
        res = fn(res, x)
    return res


def negList(xs: Iterable[float]) -> List[float]:
    """
    Return a list where each element is the negation of the corresponding input.

    Args:
        xs (Iterable[float]): Input iterable of floats.

    Returns:
        List[float]: List of negated values.
    """
    return map_fn(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> List[float]:
    """
    Add corresponding elements of two lists.

    Args:
        xs (Iterable[float]): First list.
        ys (Iterable[float]): Second list.

    Returns:
        List[float]: List of elementwise sums.
    """
    return zipWith(add, xs, ys)


def sum_list(xs: Iterable[float]) -> float:
    """
    Sum all values in the input list.

    Args:
        xs (Iterable[float]): List of numbers.

    Returns:
        float: Sum of all elements in the list.
    """
    return float(reduce_fn(add, xs, 0.0))


def prod(xs: Iterable[float]) -> float:
    """
    Return the product of all elements in the list.

    Args:
        xs (Iterable[float]): List of numbers.

    Returns:
        float: Product of all elements.
    """
    return float(reduce_fn(mul, xs, 1.0))

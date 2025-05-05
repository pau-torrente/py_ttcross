import numpy as np
import numba as nb

# Some example functions, all of the compilable with numba -> Performance improvement is immense


def log_of_sum(x: np.ndarray) -> float:
    return np.log(np.sum(x))


def slater(x: np.ndarray) -> float:
    return np.exp(-np.linalg.norm(x)) / np.linalg.norm(x)


def gaussian(x: np.ndarray) -> float:
    return np.exp(-(np.sum(x**2)))


def inverse_mod(x: np.ndarray) -> float:
    return 1 / np.sum(x**2)


# More example functions


def test_function1(x: np.ndarray) -> np.ndarray:
    return np.log(np.prod(x))


def test_function2(x: np.ndarray) -> float:
    return np.sin(sum(x)) * np.prod(x * np.exp(-x)) * (3 * np.linalg.norm(x) + 1 + x[0])


def test_function3(x: np.ndarray) -> float:
    return np.sin(sum(x)) * np.prod(np.exp(-x))


def test_function4(x: np.ndarray) -> float:
    return 1 / (1 + np.sum(x))


def test_function5(x: np.ndarray) -> np.ndarray:
    return np.prod(np.sin(x) / 2)

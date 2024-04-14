import numpy as np
import numba as nb


@nb.njit()  # -> Especially important for cases where the function contains loops
def func_template(x: np.ndarray, num_variables: int = None) -> complex:
    # THIS IS OPTIONAL, AND MIGHT SLOW DOWN THE FUNCTION
    if num_variables is not None and len(x) != num_variables:
        raise ValueError("The number of variables does not match the number of variables in the function")

    if not isinstance(x.dtype, (np.float_, np.int_)):
        raise ValueError("The input array must contain floats or ints, not complex numbers")

    value = 0
    # ADD THE FUNCTION TO THE VALUE

    return value


def log_of_sum(x: np.ndarray, num_variables: int = None) -> float:
    return np.log(np.sum(x))


def slater(x: np.ndarray, num_variables: int = None) -> float:
    return np.exp(-np.linalg.norm(x)) / np.linalg.norm(x)


def convert_from_binary(x: np.ndarray, d: int) -> float:
    # Here we are assuming that x contains only 0s and 1s
    return np.sum([2**i * x[i] for i in range(d)])

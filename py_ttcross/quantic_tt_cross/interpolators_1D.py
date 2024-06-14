from py_ttcross.regular_tt_cross.dmrg_cross import greedy_cross, ttrc
import numpy as np
from ncon import ncon
from types import FunctionType
from abc import ABC, abstractmethod


class one_dim_function_interpolator(ABC):
    """Base one-dimensional function interpolator class. It contains the masic method to convert the problem of
    interpolating a function f(x) in a given interval with a desired number of points into a problem in a
    multidimensional bianry grid.

    Args:
        - func (FunctionType): The function to interpolate. It must be a function that takes a single float as input and
            return a single float or complex as output.
        - interval (list[float, float]): The interval in which the function must be interpolated.
        - d (int): The number of dimensions of the binary grid. This will translate into 2**d points in the interval.
        - complex_function (bool): Whether the function is complex or not. Defaults to False.
    """

    def __init__(
        self, func: FunctionType, interval: list[float, float], d: int, complex_function: bool = False
    ) -> None:
        self.d = d
        n = 2**d
        self.h = (interval[1] - interval[0]) / n
        self.interval = interval
        self.grid = np.array(
            [[0, 1] for _ in range(self.d)],
            dtype=np.float64,
        )
        self.complex_f = complex_function
        self.func = func
        self.interpolated = False

    def x(self, binary_i: np.ndarray) -> np.float_:
        """Converts the binary string into a float in the interval.

        Args:
            - binary_i (np.ndarray): The binary representation of a point x in the interval, given as a numpy array of
                0s and 1s.

        Returns:
            - np.float_: The float representation of the binary string in the interval.
        """
        i = np.sum([ip * 2**index for index, ip in enumerate(np.flip(binary_i))])

        return (i + 0.5) * self.h + self.interval[0]
        # return (i) * self.h + self.interval[0]

    def func_from_binary(self, binary_i: np.ndarray) -> np.float_:
        """Evaluate the function in a point given as a binary string.

        Args:
            - binary_i (np.ndarray): The binary representation of a point x in the interval, given as a numpy array of
                0s and 1s.

        Returns:
            - np.float_: The value of the function in the point x.
        """
        return self.func(self.x(binary_i))

    def _eval_contraction_tensors(self, x: np.float_) -> np.ndarray:
        """Method that creates the array of vectors that must be contracted into the free legs of the tensors in the
        tensor train interpolation to evaluate the function in a point x.

        Args:
            - x (np.float_): The value at which the function must be evaluated. This x must be in the interval in which
                the function was interpolated.

        Returns:
            - np.ndarray: The array of vectors that must be contracted into the free legs of the tensors in the tensor
                train to evaluate the function in the point x from the approximation.
        """
        if x == self.interval[1]:
            i = 2**self.d - 1
        else:
            i = int(np.round((x - self.interval[0]) / self.h - 0.5))

        bin_i = np.array([int(ip) for ip in np.binary_repr(i, width=self.d)], dtype=np.int_)
        return np.array([[1, 0] if bin_i[site] == 0 else [0, 1] for site in range(self.d)], dtype=np.int_)

    def eval(self, x: np.float_) -> np.float_ | np.complex_:
        """Method to evaluate the function in a point x in the interval from the tensor train interpolation

        Args:
            - x (np.float_): The value at which the function must be evaluated. This x must be in the interval in which
                the function was interpolated.

        Raises:
            - ValueError: If the function has not been interpolated yet.

        Returns:
            - np.float_ | np.complex_: The value of the function in the point x.
        """
        if not self.interpolated:
            raise ValueError("The function has not been interpolated yet")

        # Copy the interpolation tensors to avoid modifying them and create the contraction tensors for the point x
        interpolation_tensors = self.interpolation.copy()
        contr_tensors = self._eval_contraction_tensors(x)

        # Contract the tensors to evaluate the function in the point x in order going from left to right in the tensor
        # train to be as efficient as possible. Finally, return the value of the function in the point x.
        result = interpolation_tensors[0][0]
        result = ncon(
            [contr_tensors[0], result],
            [[1], [1, -1]],
        )

        for i in range(1, self.d):
            result = ncon(
                [result, interpolation_tensors[2 * i - 1]],
                [[1], [1, -1]],
            )

            result = ncon([result, interpolation_tensors[2 * i]], [[1], [1, -1, -2]])

            result = ncon(
                [contr_tensors[i], result],
                [[1], [1, -1]],
            )

        return result[0]

    @abstractmethod
    def interpolate(self, *args, **kwargs) -> None:
        """Method that call the desired interpolator to interpolate the function in the interval using a binary grid."""


class greedy_one_dim_func_interpolator(one_dim_function_interpolator):
    """Class representing a one-dimensional function interpolator that uses the ttcross greedy interpolator."""

    def __init__(
        self,
        func: FunctionType,
        interval: list[float, float],
        d: int,
        complex_function: bool,
        pivot_initialization: str = "random",
    ) -> None:
        super().__init__(func, interval, d, complex_function)
        self.pivot_init = pivot_initialization

    def interpolate(self, max_bond: int, pivot_finder_tol: float, sweeps: int) -> None:
        """Method that call the ttcross greedy interpolator to interpolate the function in the interval using a binary
        grid.

        Args:
            - max_bond (int): The max bond dimension of the MPS that will be used to interpolate the function.

            - pivot_finder_tol (float): The tolerance used in the pivot finder algorithm. Setting this to values below
            1e-10 is not recommended, as the algorithm may produce singular matrices by selecting pivots that are really
            really similar to what is already in the approximation.

            - sweeps (int): The number of sweeps that the algorithm will perform.
        """
        self.interpolator = greedy_cross(
            func=self.func_from_binary,
            num_variables=self.d,
            grid=self.grid,
            tol=pivot_finder_tol,
            max_bond=max_bond,
            sweeps=sweeps,
            is_f_complex=self.complex_f,
            pivot_initialization=self.pivot_init,
        )
        self.interpolation = self.interpolator.run()
        self.interpolated = True


class ttrc_one_dim_func_interpolator(one_dim_function_interpolator):
    """Class representing a one-dimensional function interpolator that uses the ttrc interpolator."""

    def __init__(
        self,
        func: FunctionType,
        interval: list[float, float],
        d: int,
        complex_function: bool,
        pivot_initialization: str = "random",
    ) -> None:
        super().__init__(func, interval, d, complex_function)
        self.pivot_init = pivot_initialization

    def interpolate(
        self, initial_bond_guess: int, max_bond: int, maxvol_tol: float, truncation_tol: float, sweeps: int
    ) -> None:
        """Method that call the ttrc interpolator to interpolate the function in the interval using a binary grid.

        Args:
            - max_bond (int): The max bond dimension of the MPS that will be used to interpolate the function.
            - maxvol_tol (float): The tolerance used in the maxvol algorithm. The closer to 0, the more accurate the
                interpolation will be.
            - truncation_tol (float): The tolerance used in the truncation performed in the SVD procedure. The closer to
                0, the less eigenvalues will be eliminated.
            - sweeps (int): The number of sweeps that the algorithm will perform.
        """
        self.interpolator = ttrc(
            func=self.func_from_binary,
            num_variables=self.d,
            grid=self.grid,
            maxvol_tol=maxvol_tol,
            truncation_tol=truncation_tol,
            sweeps=sweeps,
            initial_bond_guess=initial_bond_guess,
            max_bond=max_bond,
            is_f_complex=self.complex_f,
            pivot_initialization=self.pivot_init,
        )

        self.interpolation = self.interpolator.run()
        self.interpolated = True

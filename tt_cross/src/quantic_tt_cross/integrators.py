from tt_cross.src.regular_tt_cross.dmrg_cross import greedy_cross, ttrc

import numpy as np
from abc import ABC, abstractmethod
from types import FunctionType
from ncon import ncon


class qtt_cross_integrator(ABC):
    """Base class for the integrators based on the quantics-tt-cross decomposition. This class implements the base
    methods to transform the binary representation of the variables into the real values and the weights for the
    quadrature. This quadrature averages the function over the grid of the variables. More sophisticated quadrature
    rules could be implemented as a tensor train, but this is not implemented yet.

    Args:
         - func (FunctionType): The function to be integrated. Should take a numpy vector as an input and return a float
        or a complex number. In this case, the numbe compilation is not supported as it has to go through the binary
        representation of the variables.

        - num_variables (int): The number of variables of the function to be integrated.

        - intervals (np.ndarray): The intervals of integration for each variable. It should be a numpy array of shape
        (num_variables, 2), where the first column contains the lower bounds and the second column contains the upper
        bounds.

        - d (int): The number of digits to be used in the binary representation of the variables.

        - is_f_complex (bool): Whether the function to be integrated returns complex numbers or not. Defauls to False.

        - quadrature (str): The quadrature method to be used. Can be only be "Rectangular" at the moment.

        - pivot_initialization (str): The method to be used to initialize the pivot sets in the tt-cross algorithm. Can
        be "random" or "first_n". Defaults to "random".
    """

    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        intervals: list[float, float],
        d: int,
        is_f_complex: bool = False,
        quadrature: str = "Rectangular",
        pivot_initialization: str = "random",
    ) -> None:

        if quadrature != "Rectangular":
            raise NotImplementedError("Only the Rectangular quadrature is implemented")

        if len(intervals) != num_variables:
            raise ValueError("The number of intervals must match the number of variables")

        self.quad = quadrature

        self.d = d
        self.n = 2**d
        self.h = np.array([(interval[1] - interval[0]) for interval in intervals])
        self.intervals = intervals
        self.num_var = num_variables

        self.grid = np.array(
            [[0, 1] for _ in range(self.d * num_variables)],
            dtype=np.float64,
        )
        self._create_rectangular_weights()

        self.complex_f = is_f_complex
        self.func = func
        self.interpolated = False
        self.func_calls = 0
        self.pivot_init = pivot_initialization

    def _x(self, binary: np.ndarray, ind: int) -> np.float_:
        """Helper function to transform the binary representation of a variable into the real value.

        Args:
            binary (np.ndarray): The binary representation of the variable as an array of 0s and 1s.
            ind (int): Which variable is being transformed.

        Returns:
            np.float_: The real value of the variable.
        """
        i = np.sum([ip * 2**index for index, ip in enumerate(np.flip(binary))])
        return (i + 0.5) * self.h[ind] / self.n + self.intervals[ind][0]

    def x_arr(self, binary: np.ndarray) -> np.ndarray:
        """Method to transform the binary representation of all the variables into the real values.

        Args:
            binary (np.ndarray): The binary representation of all the variables as a flat array of 0s and 1s.

        Returns:
            np.ndarray: An array containing the real values of all the variables.
        """
        return np.array([self._x(binary[i * self.d : (i + 1) * self.d], i) for i in range(self.num_var)])

    def func_from_binary(self, binary: np.ndarray) -> np.float_ | np.complex_:
        """Method to evaluate the function to be integrated from the binary representation of the variables.

        Args:
            binary (np.ndarray): The binary representation of all the variables as a flat array of 0s and 1s.

        Returns:
            np.float_ | np.complex_: The value of the function at the given point.
        """
        return self.func(self.x_arr(binary))

    def _create_rectangular_weights(self):
        """Helper method to create the weights for the rectangular quadrature rule."""
        self.weights = (
            0.5
            * np.prod(self.h ** (1 / (self.num_var * self.d)))
            * np.ones((self.num_var * self.d, 2), dtype=np.float_)
        )

    @abstractmethod
    def _interpolate(self):
        """Method that calls the desired interpolator"""

    def integrate(self) -> np.float_ | np.complex_:
        """Method that computes the integral of the function from the qtt-cross approximation using a rectangular
        quadrature rule (average over the grid).

        Returns:
            np.float_ | np.complex_ : The value of the integral.
        """
        if not self.interpolated:
            tensor_train = self._interpolate()
            self.interpolated = True

        result = ncon(
            [self.weights[0], tensor_train[0][0]],
            [[1], [1, -1]],
        )

        for i in range(1, self.num_var * self.d):
            result = ncon(
                [result, tensor_train[2 * i - 1]],
                [[1], [1, -1]],
            )

            result = ncon([result, tensor_train[2 * i]], [[1], [1, -1, -2]])

            result = ncon(
                [self.weights[i], result],
                [[1], [1, -1]],
            )

        return result[0]


class ttrc_qtt_integrator(qtt_cross_integrator):
    """Class that implements the quantics-tt-cross decomposition using the TTRC algorithm to integrate a function.

    Args:
         - func (FunctionType): The function to be integrated. Should take a numpy vector as an input and return a float
        or a complex number. For speed purposes, it is highly recommended to pass a function which can be numba jit
        compiled.

        - num_variables (int): The number of variables of the function to be integrated.

        - intervals (np.ndarray): The intervals of integration for each variable. It should be a numpy array of shape
        (num_variables, 2), where the first column contains the lower bounds and the second column contains the upper
        bounds.

        - d (int): The number of digits to be used in the binary representation of the variables.

        - sweeps (int): The number of sweeps to be performed by the tt-cross algorithm.

        - initial_bond_guess (int): The initial number of pivots to be used in the ttrc algorithm as an initial crude
        approximation

        - max_bond (int): The maximum number of pivots allowed at each cross block in the ttrc algorithm.

        - maxvol_tol (float): The tolerance to be used in the pivot finding procedure of the maxvol algorithm.
        Defaults to 1e-4.

        - truncation_tol (float): The tolerance to be used in the truncation process of the ttrc algorithm after
        performing the SVD on the 2-site blocks. Defaults to 1e-10.

        - is_f_complex (bool): Whether the function to be integrated returns complex numbers or not. Defauls to False.

        - quadrature (str): The quadrature method to be used. Can be "Simpson", "Trapezoidal" or "Gauss". Defaults to
        "Trapezoidal".

        - pivot_initialization (str): The method to be used to initialize the pivot sets in the tt-cross algorithm. Can
        be "random" or "first_n". Defaults to "random".
    """

    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        intervals: np.ndarray,
        d: int,
        sweeps: int,
        initial_bond_guess: int,
        max_bond: int,
        maxvol_tol: float = 1e-4,
        truncation_tol: float = 1e-10,
        is_f_complex: bool = False,
        quadrature: str = "Rectangular",
        pivot_initialization: str = "random",
    ) -> None:

        super().__init__(
            func,
            num_variables,
            intervals,
            d,
            is_f_complex,
            quadrature,
            pivot_initialization,
        )

        self.sweeps = sweeps
        self.maxvol_tol = maxvol_tol
        self.truncation_tol = truncation_tol
        self.init_bond = initial_bond_guess
        self.maxbond = max_bond

    def _interpolate(self) -> np.ndarray:
        """Helper method to call the ttrc algorithm to perform the tt-cross decomposition of the function.

        Returns:
            np.ndarray: The array containing the qtt-cross decomposition of the function.
        """
        self.interpolator = ttrc(
            func=self.func_from_binary,
            num_variables=self.d * self.num_var,
            grid=self.grid,
            maxvol_tol=self.maxvol_tol,
            truncation_tol=self.truncation_tol,
            sweeps=self.sweeps,
            initial_bond_guess=self.init_bond,
            max_bond=self.maxbond,
            is_f_complex=self.complex_f,
            pivot_initialization=self.pivot_init,
        )

        self.interpolation = self.interpolator.run()
        self.func_calls = self.interpolator.func_calls
        return self.interpolation


class greedy_qtt_cross_integrator(qtt_cross_integrator):
    """Class that implements the quantics-tt-cross decomposition using the Greedy-Cross algorithm to integrate a function.

    Args:
         - func (FunctionType): The function to be integrated. Should take a numpy vector as an input and return a float
        or a complex number. For speed purposes, it is highly recommended to pass a function which can be numba jit
        compiled.

        - num_variables (int): The number of variables of the function to be integrated.

        - intervals (np.ndarray): The intervals of integration for each variable. It should be a numpy array of shape
        (num_variables, 2), where the first column contains the lower bounds and the second column contains the upper
        bounds.

        - d (int): The number of digits to be used in the binary representation of the variables.

        - sweeps (int): The number of sweeps to be performed by the tt-cross algorithm.

        - initial_bond_guess (int): The initial number of pivots to be used in the ttrc algorithm as an initial crude
        approximation

        - max_bond (int): The maximum number of pivots allowed at each cross block in the ttrc algorithm.

        - maxvol_tol (float): The tolerance to be used in the pivot finding procedure of the maxvol algorithm.
        Defaults to 1e-4.

        - truncation_tol (float): The tolerance to be used in the truncation process of the ttrc algorithm after
        performing the SVD on the 2-site blocks. Defaults to 1e-10.

        - is_f_complex (bool): Whether the function to be integrated returns complex numbers or not. Defauls to False.

        - quadrature (str): The quadrature method to be used. Can be "Simpson", "Trapezoidal" or "Gauss". Defaults to
        "Trapezoidal".

        - pivot_initialization (str): The method to be used to initialize the pivot sets in the tt-cross algorithm. Can
        be "random" or "first_n". Defaults to "random".
    """

    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        intervals: np.ndarray,
        d: int,
        sweeps: int,
        max_bond: int,
        pivot_finder_tol: float = 1e-10,
        is_f_complex: bool = False,
        quadrature: str = "Rectangular",
        pivot_initialization="random",
    ) -> None:

        super().__init__(
            func,
            num_variables,
            intervals,
            d,
            is_f_complex,
            quadrature,
        )

        self.sweeps = sweeps
        self.tol = pivot_finder_tol
        self.maxbond = max_bond
        self.pivot_init = pivot_initialization

    def _interpolate(self) -> np.ndarray:
        """Helper method to call the greedy-cross algorithm to perform the tt-cross decomposition of the function."""
        self.interpolator = greedy_cross(
            func=self.func_from_binary,
            num_variables=self.num_var * self.d,
            grid=self.grid,
            tol=self.tol,
            max_bond=self.maxbond,
            sweeps=self.sweeps,
            is_f_complex=self.complex_f,
            pivot_initialization=self.pivot_init,
        )

        self.interpolation = self.interpolator.run()
        self.func_calls = self.interpolator.func_calls
        return self.interpolation

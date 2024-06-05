from tt_cross.src.regular_tt_cross.dmrg_cross import greedy_cross, ttrc

import numpy as np
from abc import ABC, abstractmethod
from types import FunctionType
from ncon import ncon


class qtt_cross_integrator(ABC):
    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        intervals: list[float, float],
        d: int,
        complex_function: bool = False,
        quadrature: str = "trapezoidal",
    ) -> None:

        if quadrature != "trapezoidal":
            raise NotImplementedError("Only trapezoidal quadrature is implemented")

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
        self._create_trapezoidal_weights()

        self.complex_f = complex_function
        self.func = func
        self.interpolated = False
        self.func_calls = 0

    def _x(self, binary: np.ndarray, ind: int) -> np.float_:
        i = np.sum([ip * 2**index for index, ip in enumerate(np.flip(binary))])
        return (i + 0.5) * self.h[ind] / self.n + self.intervals[ind][0]

    def x_arr(self, binary: np.ndarray) -> np.ndarray:
        return np.array([self._x(binary[i * self.d : (i + 1) * self.d], i) for i in range(self.num_var)])

    def func_from_binary(self, binary: np.ndarray) -> np.float_:
        return self.func(self.x_arr(binary))

    def _create_trapezoidal_weights(self):
        self.weights = (
            0.5
            * np.prod(self.h ** (1 / (self.num_var * self.d)))
            * np.ones((self.num_var * self.d, 2), dtype=np.float_)
        )

    @abstractmethod
    def _interpolate(self):
        """Method that calls the desired interpolator"""

    def integrate(self) -> complex:
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
        quadrature: str = "trapezoidal",
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
        self.maxvol_tol = maxvol_tol
        self.truncation_tol = truncation_tol
        self.init_bond = initial_bond_guess
        self.maxbond = max_bond

    def _interpolate(self):
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
        )

        self.interpolation = self.interpolator.run()
        self.func_calls = self.interpolator.func_calls
        return self.interpolation


class greedy_qtt_cross_integrator(qtt_cross_integrator):
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
        quadrature: str = "trapezoidal",
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

    def _interpolate(self):
        self.interpolator = greedy_cross(
            func=self.func_from_binary,
            num_variables=self.num_var * self.d,
            grid=self.grid,
            tol=self.tol,
            max_bond=self.maxbond,
            sweeps=self.sweeps,
            is_f_complex=self.complex_f,
        )

        self.interpolation = self.interpolator.run()
        self.func_calls = self.interpolator.func_calls
        return self.interpolation

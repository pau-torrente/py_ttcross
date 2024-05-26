from types import FunctionType
import numpy as np
from numpy.polynomial.legendre import leggauss
from .dmrg_cross import greedy_cross, ttrc
from abc import ABC, abstractmethod
from ncon import ncon
import time


class tt_integrator(ABC):
    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        intervals: np.ndarray,
        points_per_variable: int | list[int],
        sweeps: int,
        is_f_complex: bool,
        quadrature: str,
    ) -> None:

        self.func = func
        self.num_variables = num_variables

        if len(intervals) != num_variables and all(interval.shape != (2,) for interval in intervals):
            raise ValueError("Invalid intervals")
        self.intervals = intervals

        if isinstance(points_per_variable, int):
            self.points_per_variable = [points_per_variable] * num_variables
        elif len(points_per_variable) != num_variables:
            raise ValueError("Length of the points_per_variable list must be equal to num_variables")
        else:
            self.points_per_variable = points_per_variable

        self.sweeps = sweeps
        self.is_f_complex = is_f_complex

        if quadrature not in ["Simpson", "Trapezoidal", "Gauss"]:
            raise ValueError("Invalid quadrature method")

        self.quadrature = quadrature
        self.weights = np.ndarray(self.num_variables, dtype=object)
        self.grid = np.ndarray(self.num_variables, dtype=object)
        self._initialize_grid_and_weights()

    def _initialize_grid_and_weights(self):
        match self.quadrature:
            case "Simpson":
                self.grid = np.array(
                    [
                        np.linspace(self.intervals[i][0], self.intervals[i][1], self.points_per_variable[i])
                        for i in range(self.num_variables)
                    ]
                )

                self._create_simpson_weights()

            case "Trapezoidal":
                self.grid = np.array(
                    [
                        np.linspace(self.intervals[i][0], self.intervals[i][1], self.points_per_variable[i])
                        for i in range(self.num_variables)
                    ]
                )

                self._create_trapezoidal_weights()

            case "Gauss":
                for i in range(self.num_variables):
                    b = self.intervals[i][1]
                    a = self.intervals[i][0]
                    points, weights = leggauss(self.points_per_variable[i])
                    self.grid[i] = 0.5 * (b - a) * points + 0.5 * (b + a)
                    self.weights[i] = 0.5 * (b - a) * weights

            case _:
                raise NotImplementedError("This quadrature method is not implemented")

    def _create_trapezoidal_weights(self):
        for k in range(self.num_variables):
            h = self.grid[k][1] - self.grid[k][0]
            self.weights[k] = np.zeros(self.points_per_variable[k])
            self.weights[k][0] = 1
            self.weights[k][-1] = 1
            self.weights[k][1:-1] = 2
            self.weights[k] *= h / 2

    def _create_simpson_weights(self):
        for k in range(self.num_variables):
            if self.points_per_variable[k] % 2 == 0:
                raise ValueError(
                    f"Number of points per variable must be odd for Simpson quadrature. In the variable {k} it is {self.points_per_variable[k]}"
                )

            h = self.grid[k][1] - self.grid[k][0]

            self.weights[k] = np.zeros(self.points_per_variable[k])
            self.weights[k][0] = 1
            self.weights[k][-1] = 1

            if self.points_per_variable[k] > 3:
                self.weights[k][1:-1:2] = 4
                self.weights[k][2:-1:2] = 2
            else:
                self.weights[k][1] = 4

            self.weights[k] *= h / 3

    @abstractmethod
    def _interpolate(self):
        """Method that calls the desired interpolator."""

    def integrate(self) -> float | complex:
        """Method to integrate the function using the quadrature method specified in the constructor.

        Returns:
            float | complex: The value of the integral of the function.
        """
        tensor_train = self._interpolate()

        result = ncon(
            [self.weights[0], tensor_train[0][0]],
            [[1], [1, -1]],
        )

        for i in range(1, self.num_variables):
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


class ttrc_integrator(tt_integrator):
    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        intervals: np.ndarray,
        points_per_variable: int | list[int],
        sweeps: int,
        initial_bond_guess: int,
        max_bond: int,
        maxvol_tol: float = 1e-4,
        truncation_tol: float = 1e-10,
        is_f_complex: bool = False,
        quadrature: str = "Trapezoidal",
    ) -> None:

        super().__init__(
            func,
            num_variables,
            intervals,
            points_per_variable,
            sweeps,
            is_f_complex,
            quadrature,
        )

        self.maxvol_tol = maxvol_tol
        self.truncation_tol = truncation_tol
        self.init_bond = initial_bond_guess
        self.maxbond = max_bond

    def _interpolate(self):
        self.interpolator = ttrc(
            func=self.func,
            num_variables=self.num_variables,
            grid=self.grid,
            maxvol_tol=self.maxvol_tol,
            truncation_tol=self.truncation_tol,
            sweeps=self.sweeps,
            initial_bond_guess=self.init_bond,
            max_bond=self.maxbond,
            is_f_complex=self.is_f_complex,
        )

        self.interpolation = self.interpolator.run()
        return self.interpolation


class greedy_cross_integrator(tt_integrator):
    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        intervals: np.ndarray,
        points_per_variable: int | list[int],
        sweeps: int,
        max_bond: int,
        pivot_finder_tol: float = 1e-10,
        is_f_complex: bool = False,
        quadrature: str = "Trapezoidal",
    ) -> None:

        super().__init__(
            func,
            num_variables,
            intervals,
            points_per_variable,
            sweeps,
            is_f_complex,
            quadrature,
        )

        self.tol = pivot_finder_tol
        self.maxbond = max_bond

    def _interpolate(self):
        self.interpolator = greedy_cross(
            func=self.func,
            num_variables=self.num_variables,
            grid=self.grid,
            tol=self.tol,
            max_bond=self.maxbond,
            sweeps=self.sweeps,
            is_f_complex=self.is_f_complex,
        )

        self.interpolation = self.interpolator.run()
        return self.interpolation


class tracked_ttrc_integrator(ttrc, tt_integrator):
    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        intervals: np.ndarray,
        points_per_variable: int | list[int],
        sweeps: int,
        initial_bond_guess: int,
        max_bond: int,
        maxvol_tol: float = 1e-4,
        truncation_tol: float = 1e-10,
        is_f_complex: bool = False,
        quadrature: str = "Trapezoidal",
    ) -> None:

        self.func = func
        self.num_variables = num_variables

        if len(intervals) != num_variables and all(interval.shape != (2,) for interval in intervals):
            raise ValueError("Invalid intervals")
        self.intervals = intervals

        if isinstance(points_per_variable, int):
            self.points_per_variable = [points_per_variable] * num_variables
        elif len(points_per_variable) != num_variables:
            raise ValueError("Length of the points_per_variable list must be equal to num_variables")
        else:
            self.points_per_variable = points_per_variable

        self.sweeps = sweeps
        self.is_f_complex = is_f_complex

        if quadrature not in ["Simpson", "Trapezoidal", "Gauss"]:
            raise ValueError("Invalid quadrature method")

        self.quadrature = quadrature
        self.weights = np.ndarray(self.num_variables, dtype=object)
        self.grid = np.ndarray(self.num_variables, dtype=object)
        self._initialize_grid_and_weights()
        self.func_calls = 0
        self.evolution_dict = {}

        super().__init__(
            func=func,
            num_variables=num_variables,
            grid=self.grid,
            maxvol_tol=maxvol_tol,
            truncation_tol=truncation_tol,
            sweeps=sweeps,
            initial_bond_guess=initial_bond_guess,
            max_bond=max_bond,
            is_f_complex=is_f_complex,
        )

    def compute_superblock_tensor(self, site: int) -> np.ndarray:
        tensor = super().compute_superblock_tensor(site)

        self.func_calls += np.prod(tensor.shape)

        return tensor

    def full_sweep(self) -> None:
        super().full_sweep()

        try:
            mps = np.ndarray(2 * self.num_variables - 1, dtype=np.ndarray)

            for site in range(self.num_variables - 1):
                mps[2 * site] = self.compute_single_site_tensor(site)
                mps[2 * site + 1] = self.compute_cross_blocks(site)

            mps[-1] = self.compute_single_site_tensor(self.num_variables - 1)

            result = ncon(
                [self.weights[0], mps[0][0]],
                [[1], [1, -1]],
            )

            for i in range(1, self.num_variables):
                result = ncon(
                    [result, mps[2 * i - 1]],
                    [[1], [1, -1]],
                )

                result = ncon([result, mps[2 * i]], [[1], [1, -1, -2]])

                result = ncon(
                    [self.weights[i], result],
                    [[1], [1, -1]],
                )

            self.evolution_dict[self.func_calls] = result[0]

        except:
            pass

    def _interpolate(self):
        return self.run()


class tracked_greedycross_integrator(greedy_cross, tt_integrator):
    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        intervals: np.ndarray,
        points_per_variable: int | list[int],
        sweeps: int,
        max_bond: int,
        pivot_finder_tol: float = 1e-10,
        is_f_complex: bool = False,
        quadrature: str = "Trapezoidal",
    ) -> None:

        self.func = func
        self.num_variables = num_variables

        if len(intervals) != num_variables and all(interval.shape != (2,) for interval in intervals):
            raise ValueError("Invalid intervals")
        self.intervals = intervals

        if isinstance(points_per_variable, int):
            self.points_per_variable = [points_per_variable] * num_variables
        elif len(points_per_variable) != num_variables:
            raise ValueError("Length of the points_per_variable list must be equal to num_variables")
        else:
            self.points_per_variable = points_per_variable

        self.sweeps = sweeps
        self.is_f_complex = is_f_complex

        if quadrature not in ["Simpson", "Trapezoidal", "Gauss"]:
            raise ValueError("Invalid quadrature method")

        self.quadrature = quadrature
        self.weights = np.ndarray(self.num_variables, dtype=object)
        self.grid = np.ndarray(self.num_variables, dtype=object)
        self._initialize_grid_and_weights()
        self.func_calls = 0
        self.evolution_dict = {}

        super().__init__(
            func=func,
            num_variables=num_variables,
            grid=self.grid,
            tol=pivot_finder_tol,
            max_bond=max_bond,
            sweeps=sweeps,
            is_f_complex=is_f_complex,
        )

    def compute_superblock_tensor(self, site: int) -> np.ndarray:
        tensor = super().compute_superblock_tensor(site)

        self.func_calls += np.prod(tensor.shape)

        return tensor

    def index_update(self, site: int):
        super().index_update(site)

        mps = np.ndarray(2 * self.num_variables - 1, dtype=np.ndarray)

        for site in range(self.num_variables - 1):
            mps[2 * site] = self.compute_single_site_tensor(site)
            mps[2 * site + 1] = self.compute_cross_blocks(site)

        mps[-1] = self.compute_single_site_tensor(self.num_variables - 1)

        result = ncon(
            [self.weights[0], mps[0][0]],
            [[1], [1, -1]],
        )

        for i in range(1, self.num_variables):
            result = ncon(
                [result, mps[2 * i - 1]],
                [[1], [1, -1]],
            )

            result = ncon([result, mps[2 * i]], [[1], [1, -1, -2]])

            result = ncon(
                [self.weights[i], result],
                [[1], [1, -1]],
            )

        self.evolution_dict[self.func_calls] = result[0]

    def _interpolate(self):
        return self.run()


class simple_mc:
    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        intervals: np.ndarray,
        points_per_variable: int | list[int],
    ):
        self.func = func
        self.num_variables = num_variables

        if len(intervals) != num_variables and all(interval.shape != (2,) for interval in intervals):
            raise ValueError("Invalid intervals")
        self.intervals = intervals

        if isinstance(points_per_variable, int):
            self.points_per_variable = [points_per_variable] * num_variables
        elif len(points_per_variable) != num_variables:
            raise ValueError("Length of the points_per_variable list must be equal to num_variables")
        else:
            self.points_per_variable = points_per_variable

        self.normalization = np.prod([interval[1] - interval[0] for interval in self.intervals])
        self.grid = np.array(
            [
                np.linspace(self.intervals[i][0], self.intervals[i][1], self.points_per_variable[i])
                for i in range(self.num_variables)
            ]
        )

        self.sum = 0
        self.evolution = []

    def sample(self):

        # TODO Finish this sample method
        point = self.grid[
            np.arange(self.num_variables), np.random.randint(0, self.points_per_variable, self.num_variables)
        ]

        return self.func(point)

    def integrate(self, n_samples: int):
        for n in range(n_samples):
            self.sum += self.sample()
            self.integral = self.sum * self.normalization / (n + 1)
            self.evolution.append(self.integral)
        return self.integral

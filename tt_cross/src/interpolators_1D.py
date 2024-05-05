from .dmrg_cross import greedy_cross, ttrc
import numpy as np
from ncon import ncon
from types import FunctionType


class one_dim_function_interpolator:
    def __init__(self, func: FunctionType, interval: list[float, float], d: int, complex_function: bool) -> None:
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
        i = np.sum([ip * 2**index for index, ip in enumerate(np.flip(binary_i))])
        return (i + 0.5) * self.h + self.interval[0]

    def func_from_binary(self, binary_i: np.ndarray) -> np.float_:
        return self.func(self.x(binary_i))

    def interpolate(self, max_bond: int, tol: float, sweeps: int) -> None:
        self.interpolator = greedy_cross(
            func=self.func_from_binary,
            num_variables=self.d,
            grid=self.grid,
            tol=tol,
            max_bond=max_bond,
            sweeps=sweeps,
            is_f_complex=self.complex_f,
        )
        self.interpolation = self.interpolator.run()
        self.interpolated = True

    def _eval_contraction_tensors(self, x: np.float_) -> np.ndarray:
        i = int(np.round((x - self.interval[0]) / self.h - 0.5))
        bin_i = np.array([int(ip) for ip in np.binary_repr(i, width=self.d)], dtype=np.int_)
        return np.array([[1, 0] if bin_i[site] == 0 else [0, 1] for site in range(self.d)], dtype=np.int_)

    def eval(self, x: np.float_) -> np.float_:
        if not self.interpolated:
            raise ValueError("The function has not been interpolated yet")

        interpolation_tensors = self.interpolation.copy()
        contr_tensors = self._eval_contraction_tensors(x)

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


class greedy_one_dim_func_interpolator(one_dim_function_interpolator):
    def __init__(self, func: FunctionType, interval: list[float, float], d: int, complex_function: bool) -> None:
        super().__init__(func, interval, d, complex_function)

    def interpolate(self, max_bond: int, pivot_finder_tol: float, sweeps: int) -> None:
        self.interpolator = greedy_cross(
            func=self.func_from_binary,
            num_variables=self.d,
            grid=self.grid,
            tol=pivot_finder_tol,
            max_bond=max_bond,
            sweeps=sweeps,
            is_f_complex=self.complex_f,
        )
        self.interpolation = self.interpolator.run()
        self.interpolated = True


class ttrc_one_dim_func_interpolator(one_dim_function_interpolator):
    def __init__(self, func: FunctionType, interval: list[float, float], d: int, complex_function: bool) -> None:
        super().__init__(func, interval, d, complex_function)

    def interpolate(self, max_bond: int, maxvol_tol: float, truncation_tol: float, sweeps: int) -> None:
        self.interpolator = ttrc(
            func=self.func_from_binary,
            num_variables=self.d,
            grid=self.grid,
            maxvol_tol=maxvol_tol,
            truncation_tol=truncation_tol,
            sweeps=sweeps,
            initial_bond_guess=max_bond,
            is_f_complex=self.complex_f,
        )

        self.interpolation = self.interpolator.run()
        self.interpolated = True

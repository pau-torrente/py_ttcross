import numpy as np
import numba as nb
from .maxvol import greedy_pivot_finder, maxvol, py_maxvol
from types import FunctionType
from ncon import ncon


class tt_interpolator:
    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        grid: np.ndarray,
        tol: float,
        sweeps: int,
        is_f_complex=False,
    ):
        self.func = func
        if len(grid) != num_variables:
            raise ValueError("The grid must have the same number of dimensions as the number of variables")
        self.num_variables = num_variables
        self.grid = grid
        self.tol = tol
        self.sweeps = sweeps

        self.f_type = np.complex128 if is_f_complex else np.float64

        self.bonds = np.ndarray(self.num_variables + 1, dtype=np.ndarray)
        self.bonds[0] = 1
        self.bonds[-1] = 1

    def _obtain_superblock_total_indices(self, site: int, compute_index_pos: bool = True):
        # OLD VERSION WITHOUT OLD INDEX INDICES

        # total_indices_left = []
        # total_indices_right = []
        # if site == 0:
        #     for i in self.grid[site]:
        #         total_indices_left.append([i])
        # else:
        #     for I_1 in self.i[site + 1 - 1]:
        #         for i in self.grid[site]:
        #             total_indices_left.append(np.concatenate((I_1, [i])))

        # if site == self.num_variables - 2:
        #     for j in self.grid[site]:
        #         total_indices_right.append([j])
        # else:
        #     for J_1 in self.j[site + 1]:
        #         for j in self.grid[site + 1]:
        #             total_indices_right.append(np.concatenate(([j], J_1)))

        # return np.array(total_indices_left), np.array(total_indices_right)

        # TODO: np.where is very slow, stack overflow users suggest using hash maps to speed up the process

        total_indices_left = []
        total_indices_right = []

        current_indices_left_pos = []
        current_indices_right_pos = []

        if site == 0:
            total_indices_left = np.array([self.grid[site].copy()], dtype=object).T
            if compute_index_pos:
                for current_best_pivot in self.i[site + 1]:
                    current_indices_left_pos.append(
                        np.where(np.all(total_indices_left == current_best_pivot, axis=1))[0][0]
                    )
        else:
            arrL_repeated = np.repeat(self.i[site + 1 - 1], self.grid[site].shape[0], axis=0)
            arrR_tiled = np.tile(self.grid[site], (1, self.i[site + 1 - 1].shape[0])).T

            total_indices_left = np.concatenate((arrL_repeated, arrR_tiled), axis=1)

            if compute_index_pos:
                for current_best_pivot in self.i[site + 1]:
                    current_indices_left_pos.append(
                        np.where(np.all(total_indices_left == current_best_pivot, axis=1))[0][0]
                    )

        if site == self.num_variables - 2:
            total_indices_right = np.array([self.grid[site + 1].copy()], dtype=object).T

            if compute_index_pos:
                for current_best_pivot in self.j[site]:
                    current_indices_right_pos.append(
                        np.where(np.all(total_indices_right == current_best_pivot, axis=1))[0][0]
                    )

        else:
            arrL_tiled = np.tile(self.grid[site + 1], (1, self.j[site + 1].shape[0])).T
            arrR_repeated = np.repeat(self.j[site + 1], self.grid[site + 1].shape[0], axis=0)

            total_indices_right = np.concatenate((arrL_tiled, arrR_repeated), axis=1)

            if compute_index_pos:
                for current_best_pivot in self.j[site]:
                    current_indices_right_pos.append(
                        np.where(np.all(total_indices_right == current_best_pivot, axis=1))[0][0]
                    )

        if not compute_index_pos:
            return total_indices_left, total_indices_right

        return total_indices_left, total_indices_right, current_indices_left_pos, current_indices_right_pos

    def compute_single_site_tensor(self, site):
        tensor = np.ndarray((len(self.i[site + 1 - 1]), len(self.grid[site]), len(self.j[site])), dtype=self.f_type)

        for s, left in enumerate(self.i[site + 1 - 1]):
            for k, right in enumerate(self.j[site]):
                for m, i in enumerate(self.grid[site]):

                    if site == 0:
                        point = np.concatenate(([i], right))
                    elif site == self.num_variables - 1:
                        point = np.concatenate((left, [i]))
                    else:
                        point = np.concatenate((left, [i], right))

                    tensor[s, m, k] = self.func(point)

        return tensor

    def compute_cross_blocks(self, site):
        if len(self.i[site + 1]) != len(self.j[site]):
            raise ValueError("The left and right indexes must have the same dimension")

        block = np.ndarray((len(self.i[site + 1]), len(self.j[site])), dtype=self.f_type)

        for s, left in enumerate(self.i[site + 1]):
            for k, right in enumerate(self.j[site]):
                block[s, k] = self.func(np.concatenate((left, right)))

        inv_block = np.linalg.inv(block)

        return inv_block

    # TODO Loops are probably not the best idea here, but we can optimize them later (use numba here)
    def compute_superblock_tensor(self, site):
        tensor = np.ndarray(
            (len(self.i[site + 1 - 1]), len(self.grid[site]), len(self.grid[site + 1]), len(self.j[site + 1])),
            dtype=self.f_type,
        )

        for s, left in enumerate(self.i[site + 1 - 1]):
            for k, right in enumerate(self.j[site + 1]):
                for m, i in enumerate(self.grid[site]):
                    for n, j in enumerate(self.grid[site + 1]):

                        if site == 0:
                            point = np.concatenate(([i], [j], right))
                        elif site == self.num_variables - 1:
                            point = np.concatenate((left, [i], [j]))
                        else:
                            point = np.concatenate((left, [i], [j], right))

                        tensor[s, m, n, k] = self.func(point)

        return tensor


class ttrc(tt_interpolator):
    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        grid: np.ndarray,
        maxvol_tol: float,
        truncation_tol: float,
        sweeps: int,
        initial_bond_guess: int,
        is_f_complex=False,
    ):
        super().__init__(func, num_variables, grid, maxvol_tol, sweeps, is_f_complex)

        if initial_bond_guess > max(grid.shape):
            initial_bond_guess = max(grid.shape)
            raise Warning(
                "The initial bond guess is larger than the maximum grid size. Max bond gets initialized to the maximum of grid size instead of given initial_bond_guess."
            )

        else:
            self.max_bond = initial_bond_guess

        self._create_initial_arrays()
        self.trunctol = truncation_tol

    def _create_initial_index_sets(self):
        self.i = np.ndarray(self.num_variables + 1, dtype=object)
        self.i[0] = np.array([[1]])
        self.i[1] = np.array(
            [np.random.choice(self.grid[0], size=min(len(self.grid[0]), self.max_bond), replace=False)]
        ).T
        for k in range(2, self.num_variables + 1):
            current_index = np.array(
                [np.random.choice(self.grid[k - 1], size=min(len(self.grid[k - 1]), self.max_bond), replace=False)]
            ).T
            if len(current_index) <= len(self.i[k - 1]):
                previous_selected_rows = np.random.choice(len(self.i[k - 1]), size=len(current_index), replace=False)
                previous_indices = self.i[k - 1][previous_selected_rows]
                self.i[k] = np.column_stack((previous_indices, current_index))
            else:
                times = len(current_index[0]) // len(self.i[k - 1][0])
                previous_choice = np.row_stack([self.i[k - 1] for _ in range(times + 1)])
                self.i[k] = np.column_stack((previous_choice[: len(current_index)], current_index))

        # =======================================================================================================

        self.j = np.ndarray(self.num_variables, dtype=np.ndarray)
        self.j[-1] = np.array([[1]])
        self.j[-2] = np.array(
            [np.random.choice(self.grid[-1], size=min(len(self.grid[-1]), self.max_bond), replace=False)]
        ).T

        for k in range(-3, -self.num_variables - 1, -1):
            current_index = np.array(
                [np.random.choice(self.grid[k + 1], size=min(len(self.grid[k + 1]), self.max_bond), replace=False)]
            ).T

            if len(current_index) <= len(self.j[k + 1]):
                selected_columns = np.random.choice(len(self.j[k + 1]), size=len(current_index), replace=False)
                previous_choice = self.j[k + 1][selected_columns]
                self.j[k] = np.column_stack((current_index, previous_choice))
            else:
                times = len(current_index) // len(self.j[k + 1])
                previous_choice = np.column_stack(([self.j[k + 1] for _ in range(times + 1)]))
                self.j[k] = np.column_stack((current_index, previous_choice[: len(current_index)]))

    def _create_initial_bond_dimensions(self):
        self.bonds = np.ndarray(self.num_variables - 1, dtype=int)
        for k in range(self.num_variables - 1):
            if self.i[k + 1].shape[0] != self.j[k].shape[0]:
                raise ValueError("Initial left and right indexes should have the same dimension")
            self.bonds[k - 1] = self.i[k + 1].shape[0]

    def _create_initial_arrays(self):
        self.p = np.ndarray(self.num_variables + 1, dtype=np.ndarray)
        self.p[0] = np.array([[1]], dtype=np.float_)
        self.p[-1] = np.array([[1]], dtype=np.float_)
        self.r = np.array([[1]], dtype=np.float_)

        self._create_initial_index_sets()
        self._create_initial_bond_dimensions()

        self.b = np.ndarray(self.num_variables, dtype=np.ndarray)
        self.b[0] = self.compute_single_site_tensor(0)

        for i in range(1, self.num_variables):
            cross_block = self.compute_cross_blocks(i - 1)
            site_block = self.compute_single_site_tensor(i)
            self.b[i] = ncon([cross_block, site_block], [[-1, 1], [1, -2, -3]])

    # =======================================================================================================

    # TODO FINISH UPDATE AND SWEEP METHODS IN TTRC WITH FULL MAXVOL

    # =======================================================================================================

    def initial_sweep(self):
        for pos in range(self.num_variables - 1, 0, -1):
            rk_1 = self.b[pos].shape[0]
            nk = self.b[pos].shape[1]
            rk = self.b[pos].shape[2]

            self.b[pos] = ncon([self.b[pos], self.r], [[-1, -2, 1], [1, -3]])

            c = np.reshape(self.b[pos], (rk_1, nk * rk))
            q_T, r_T = np.linalg.qr(c.T)

            q = q_T.T
            self.r = r_T.T

            self.b[pos] = np.reshape(q, (rk_1, nk, rk))

            q = np.reshape(q, (rk_1 * nk, rk))

            q = q @ self.p[pos + 1]

            q = np.reshape(q, (rk_1, nk * rk))

            _, J_k_1_expanded = self._obtain_superblock_total_indices(site=pos - 1, compute_index_pos=False)

            best_indices, self.j[pos - 1] = py_maxvol(A=q, full_index_set=J_k_1_expanded, tol=self.tol, max_iters=1000)

            self.p[pos] = q[:, best_indices]

        self.b[0] = ncon([self.b[0], self.r], [[-1, -2, 1], [1, -3]])

    def check_convergence(self, C: np.ndarray, site: int):
        approx_superblock = ncon([self.b[site], self.b[site + 1]], [[-1, -2, 1], [1, -3, -4]])
        err = np.linalg.norm(C - approx_superblock)
        bound = self.trunctol * np.linalg.norm(C) / np.sqrt(self.num_variables - 1)
        self.converged = err < bound

    def left_right_update(self, site):

        C = self.compute_superblock_tensor(site)

        r_left = C.shape[0]
        r_right = C.shape[3]

        inv_prev_p = np.linalg.inv(self.p[site])
        inv_post_p = np.linalg.inv(self.p[site + 2])

        C = ncon(
            [inv_prev_p, C, inv_post_p],
            [[-1, 1], [1, -2, -3, 2], [2, -4]],
        )

        self.check_convergence(C, site)

        C_reshaped = np.reshape(C, (r_left * len(self.grid[site]), len(self.grid[site + 1]) * r_right))

        utemp, s_temp, vtemp = np.linalg.svd(C_reshaped, full_matrices=True)

        stemp_cumsum = np.cumsum(s_temp)
        chitemp = min(np.argmax(stemp_cumsum > (1.0 - self.trunctol) * stemp_cumsum[-1]) + 1, self.max_bond)

        utemp = utemp[:, :chitemp]
        vtemp = vtemp[:chitemp, :]
        stemp = np.diag(s_temp[:chitemp])

        self.b[site] = np.reshape(utemp, (r_left, len(self.grid[site]), chitemp))
        right_tensor = ncon([stemp, vtemp], [[-1, 1], [1, -2]])
        self.b[site + 1] = np.reshape(right_tensor, (chitemp, len(self.grid[site + 1]), r_right))

        self.bonds[site] = chitemp

        maxvol_matrix = ncon([self.p[site], self.b[site]], [[-1, 1], [1, -2, -3]])
        maxvol_matrix = np.reshape(maxvol_matrix, (r_left * len(self.grid[site]), chitemp))

        I_k_1_expanded, _ = self._obtain_superblock_total_indices(site, compute_index_pos=False)

        best_indices, self.i[site + 1] = py_maxvol(
            A=maxvol_matrix, full_index_set=I_k_1_expanded, tol=self.tol, max_iters=1000
        )

        self.p[site + 1] = maxvol_matrix[best_indices]

    def right_left_update(self, site):

        C = self.compute_superblock_tensor(site)
        r_left = C.shape[0]
        r_right = C.shape[3]

        inv_prev_p = np.linalg.inv(self.p[site])
        inv_post_p = np.linalg.inv(self.p[site + 2])

        C = ncon(
            [inv_prev_p, C, inv_post_p],
            [[-1, 1], [1, -2, -3, 2], [2, -4]],
        )

        self.check_convergence(C, site)

        C_reshaped = np.reshape(C, (r_left * len(self.grid[site]), len(self.grid[site + 1]) * r_right))

        utemp, s_temp, vtemp = np.linalg.svd(C_reshaped, full_matrices=True)

        stemp_cumsum = np.cumsum(s_temp)
        chitemp = min(np.argmax(stemp_cumsum > (1 - self.trunctol) * stemp_cumsum[-1]) + 1, self.max_bond)

        utemp = utemp[:, :chitemp]
        vtemp = vtemp[:chitemp, :]
        stemp = np.diag(s_temp[:chitemp])

        left_tensor = ncon([utemp, stemp], [[-1, 1], [1, -2]])
        self.b[site] = np.reshape(left_tensor, (r_left, len(self.grid[site]), chitemp))
        self.b[site + 1] = np.reshape(vtemp, (chitemp, len(self.grid[site + 1]), r_right))

        self.bonds[site] = chitemp

        maxvol_matrix = ncon([self.b[site + 1], self.p[site + 2]], [[-1, -2, 1], [1, -3]])
        maxvol_matrix = np.reshape(maxvol_matrix, (chitemp, len(self.grid[site + 1]) * r_right))

        _, J_k_1_expanded = self._obtain_superblock_total_indices(site, compute_index_pos=False)
        best_indices, self.j[site] = py_maxvol(A=vtemp, full_index_set=J_k_1_expanded, tol=self.tol, max_iters=1000)

        self.p[site + 1] = vtemp[:, best_indices]

    def full_sweep(self):
        for site in range(self.num_variables - 1):
            self.left_right_update(site)

        for site in range(self.num_variables - 2, -1, -1):
            self.right_left_update(site)

    def run(self):
        self.converged = False
        self.initial_sweep()
        for sweep in range(self.sweeps):
            print("Sweep", sweep + 1)
            if self.converged:
                print(f"COnverged at sweep {sweep}")
                break
            self.full_sweep()

        mps = np.ndarray(2 * self.num_variables - 1, dtype=np.ndarray)

        for site in range(self.num_variables - 1):
            mps[2 * site] = self.compute_single_site_tensor(site)
            mps[2 * site + 1] = self.compute_cross_blocks(site)

        mps[-1] = self.compute_single_site_tensor(self.num_variables - 1)
        return mps


class greedy_cross(tt_interpolator):
    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        grid: np.ndarray,
        tol: float,
        max_bond: int,
        sweeps: int,
        is_f_complex=False,
    ):
        super().__init__(func, num_variables, grid, tol, sweeps, is_f_complex)
        self.max_bond = max_bond
        self._create_initial_index_sets()
        self._create_initial_bonds()
        self.error = []

    def _create_initial_index_sets(self):
        self.i = np.ndarray(self.num_variables + 1, dtype=object)
        self.i[0] = np.array([[1.0]])
        self.i[1] = np.array([[np.random.choice(self.grid[0])]])
        for i in range(2, self.num_variables + 1):
            current_index = np.array([np.random.choice(self.grid[i - 1])])
            self.i[i] = np.column_stack((self.i[i - 1], current_index))

        self.j = np.ndarray(self.num_variables, dtype=object)
        self.j[-1] = np.array([[1.0]])
        self.j[-2] = np.array([[np.random.choice(self.grid[-1])]])
        for i in range(-3, -self.num_variables - 1, -1):
            current_index = np.array([np.random.choice(self.grid[i + 1])])
            self.j[i] = np.column_stack((current_index, self.j[i + 1]))

    def _create_initial_index_sets(self):
        self.i = np.ndarray(self.num_variables + 1, dtype=object)
        self.i[0] = np.array([[1.0]])
        self.i[1] = np.array([[self.grid[0][0]]])
        for i in range(2, self.num_variables + 1):
            current_index = np.array([self.grid[i - 1][0]])
            self.i[i] = np.column_stack((self.i[i - 1], current_index))

        self.j = np.ndarray(self.num_variables, dtype=object)
        self.j[-1] = np.array([[1.0]])
        self.j[-2] = np.array([[self.grid[-1][0]]])
        for i in range(-3, -self.num_variables - 1, -1):
            current_index = np.array([self.grid[i + 1][0]])
            self.j[i] = np.column_stack((current_index, self.j[i + 1]))

    def _create_initial_bonds(self):
        self.bonds = np.ones(self.num_variables - 1, dtype=int)

    def index_update(self, site: int):
        if self.bonds[site] == self.max_bond:
            return

        superblock_tensor = self.compute_superblock_tensor(site)
        superblock_tensor = np.reshape(
            superblock_tensor,
            (
                superblock_tensor.shape[0] * superblock_tensor.shape[1],
                superblock_tensor.shape[2] * superblock_tensor.shape[3],
            ),
        )

        I_1i, J_1j, current_I_pos, current_J_pos = self._obtain_superblock_total_indices(site=site)
        new_I, new_J, rk, _, error = greedy_pivot_finder(
            superblock_tensor,
            I_1i,
            current_I_pos,
            J_1j,
            current_J_pos,
            tol=self.tol,
        )

        self.error.append(error)
        self.i[site + 1] = new_I
        self.j[site] = new_J
        self.bonds[site] = rk

    def full_sweep(self):
        for site in range(self.num_variables - 1):
            self.index_update(site)

        for site in range(self.num_variables - 2, -1, -1):
            self.index_update(site)

    def run(self):
        for _ in range(self.sweeps):
            pre_sweep_bonds = self.bonds.copy()
            self.full_sweep()
            if np.array_equal(pre_sweep_bonds, self.bonds):
                break

        mps = np.ndarray(2 * self.num_variables - 1, dtype=np.ndarray)

        for site in range(self.num_variables - 1):
            mps[2 * site] = self.compute_single_site_tensor(site)
            mps[2 * site + 1] = self.compute_cross_blocks(site)

        mps[-1] = self.compute_single_site_tensor(self.num_variables - 1)

        return mps

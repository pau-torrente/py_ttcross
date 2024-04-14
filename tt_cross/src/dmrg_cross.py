import numpy as np
import numba as nb
from .maxvol import greedy_pivot_finder, maxvol
from types import FunctionType


class tt_integrator:
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

    def _obtain_superblock_total_indices(self, site: int):
        total_indices_left = []
        total_indices_right = []
        if site == 0:
            for i in self.grid[site]:
                total_indices_left.append([i])
        else:
            for i_1 in self.i[site + 1 - 1]:
                for i in self.grid[site]:
                    total_indices_left.append(np.concatenate((i_1, [i])))

        if site == self.num_variables - 2:
            for j in self.grid[site]:
                total_indices_right.append([j])
        else:
            for j_1 in self.j[site + 1]:
                for j in self.grid[site + 1]:
                    total_indices_right.append(np.concatenate(([j], j_1)))

        return np.array(total_indices_left), np.array(total_indices_right)

    # TODO Loops are probably not the best idea here, but we can optimize them later (use numba here)

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


class ttrc(tt_integrator):
    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        grid: np.ndarray,
        tol: float,
        sweeps: int,
        initial_bond_guess: int,
        is_f_complex=False,
    ):
        super().__init__(func, num_variables, grid, tol, sweeps, is_f_complex)

        if initial_bond_guess > max(grid.shape):
            initial_bond_guess = max(grid.shape)
            raise Warning(
                "The initial bond guess is larger than the maximum grid size. Max bond gets initialized to the maximum of grid size instead of given initial_bond_guess."
            )

        else:
            self.max_bond = initial_bond_guess

        self._create_initial_arrays()

    def _create_initial_index_sets(self):
        self.i = np.ndarray(self.num_variables, dtype=object)
        self.i[0] = np.array([[1]])
        self.i[1] = np.array(
            [np.random.choice(self.grid[0], size=min(len(self.grid[0]), self.max_bond), replace=False)]
        )
        for k in range(2, self.num_variables):
            current_index = np.array(
                [np.random.choice(self.grid[k - 1], size=min(len(self.grid[k - 1]), self.max_bond), replace=False)]
            )
            if len(current_index) < len(self.i[k - 1]):
                previous_choice = np.array([np.random.choice(self.i[k - 1], size=len(current_index), replace=False)])
                self.i[k] = np.column_stack((previous_choice, current_index))
            else:
                times = len(current_index) // len(self.i[k - 1])
                previous_choice = np.row_stack([self.i[k - 1] for _ in range(times + 1)])
                self.i[k] = np.column_stack((previous_choice[: len(current_index)], current_index))

        # =======================================================================================================

        self.j = np.ndarray(self.num_variables, dtype=np.ndarray)
        self.j[-1] = np.array([[1]])
        self.j[-2] = np.array(
            [np.random.choice(self.grid[-1], size=min(len(self.grid[-1]), self.max_bond), replace=False)]
        )

        for k in range(-3, -self.num_variables - 1, -1):
            current_index = np.array(
                [np.random.choice(self.grid[k + 1], size=min(len(self.grid[k + 1]), self.max_bond), replace=False)]
            )

            if len(current_index) < len(self.j[k + 1]):
                previous_choice = np.array([np.random.choice(self.j[k + 1], size=len(current_index), replace=False)])
                self.j[k] = np.column_stack((current_index, previous_choice))
            else:
                times = len(current_index) // len(self.j[k + 1])
                previous_choice = np.column_stack(([self.j[k + 1] for _ in range(times + 1)]))
                self.j[k] = np.column_stack((current_index, previous_choice[: len(current_index)]))

    def _create_initial_bond_dimensions(self):
        self.bonds = np.ndarray(self.num_variables - 1, dtype=int)
        for k in range(1, self.num_variables):
            if self.i[k] != self.j[k]:
                raise ValueError("Initial left and right indexes should have the same dimension")
            self.bonds[k - 1] = len(self.i[k])

    def _create_initial_arrays(self):
        self.p = np.ndarray(self.num_variables + 1, dtype=np.ndarray)
        self.p[-1] = np.ones((1))
        self.r = np.ones((1))

        self._create_initial_index_sets()
        self._create_initial_bond_dimensions()

        self.b = np.ndarray(self.num_variables, dtype=np.ndarray)

        for i in range(self.num_variables):
            self.b[i] = self.compute_single_site_tensor(i)

    # =======================================================================================================

    # TODO FINISH UPDATE AND SWEEP METHODS IN TTRC WITH FULL MAXVOL

    # =======================================================================================================

    #   @nb.njit() -> Since we are not using ncon in this part, we can use numba
    def initial_sweep_step(self, pos: int):
        pass

        # TODO THIS PART HAS TO BE UPDATED TO CURRENT FORMAT (OUTDATED)

        # rk_1 = self.b[pos].shape[0]
        # nk = self.b[pos].shape[1]
        # rk = self.b[pos].shape[2]

        # self.b[pos] = ncon([self.b[pos], self.r], [[-1, -2, 1], [1, -3]])
        # c = np.reshape(self.b[pos], (rk_1, nk * rk))
        # q_t, r_t = np.linalg.qr(c)
        # q = q_t.T, self.r = r_t.T
        # q = np.matmul(q, self.p[pos])
        # q = np.reshape(q, (rk_1, nk * rk))
        # j_k = maxvol(q, complex=True, tol=self.tol, max_iter=1000)

        # # Does p maintain the (1,1) shape of the first and last elements?
        # self.p[pos - 1] = q[:, j_k]

        # # TODO Obtain the index values from the column index j_k
        # return j_k


class greedy_cross(tt_integrator):
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

    def _create_initial_index_sets(self):
        self.i = np.ndarray(self.num_variables, dtype=object)
        self.i[0] = np.array([[1.0]])
        self.i[1] = np.array([[np.random.choice(self.grid[0])]])
        for i in range(2, self.num_variables):
            current_index = np.array([np.random.choice(self.grid[i - 1])])
            self.i[i] = np.column_stack((self.i[i - 1], current_index))

        self.j = np.ndarray(self.num_variables, dtype=object)
        self.j[-1] = np.array([[1.0]])
        self.j[-2] = np.array([[np.random.choice(self.grid[-1])]])
        for i in range(-3, -self.num_variables - 1, -1):
            current_index = np.array([np.random.choice(self.grid[i + 1])])
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

        I_1i, J_1j = self._obtain_superblock_total_indices(site)
        new_I, new_J, rk, _ = greedy_pivot_finder(
            superblock_tensor,
            self.i[site + 1].copy(),
            I_1i,
            self.j[site].copy(),
            J_1j,
        )

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
            self.full_sweep()

        mps = np.ndarray(self.num_variables, dtype=np.ndarray)

        for site in range(self.num_variables):
            mps[site] = self.compute_single_site_tensor(site)

        return mps

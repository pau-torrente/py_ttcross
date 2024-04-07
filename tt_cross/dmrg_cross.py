import numpy as np
import numba as nb
from ncon import ncon
from .maxvol import maxvol


class ttrc:
    def __init__(self, func, num_variables, grid, tol, sweeps, initial_bond_guess, is_f_complex=False):
        self.func = func
        if len(grid.shape) != num_variables:
            raise ValueError("The grid must have the same number of dimensions as the number of variables")
        self.num_variables = num_variables
        self.grid = grid
        self.tol = tol
        self.sweeps = sweeps
        self.max_bond = initial_bond_guess
        self.bonds = np.ndarray(self.num_variables + 1, int)
        self.bonds[0] = 1
        self.bonds[-1] = 1
        self.bonds[1:-1] = self.max_bond

        self.f_type = complex if is_f_complex else np.float64

    def _create_initial_index_sets(self):

        # TODO Move the bond initialization to the _create_bond_dimensions method
        # self.bonds[0] = 1
        # self.bonds[-1] = 1

        self.i = np.ndarray(self.num_variables, dtype=np.ndarray)
        self.i[0] = self.grid[0][: min(self.grid.shape[0], self.max_bond)]

        # TODO Deduce what happens with bonds when self.max_bond > len(self.grid[0])
        for i in range(self.num_variables):
            current_index = self.grid[i][: min(self.grid.shape[i], self.max_bond)]
            self.i[i] = np.column_stack(self.i[i - 1][: len(current_index)], current_index)

        self.j = np.ndarray(self.num_variables, dtype=np.ndarray)
        self.j[-1] = self.grid[-1][: min(self.grid.shape[-1], self.max_bond)]
        for i in range(-2, -self.num_variables - 1, -1):
            current_index = self.grid[i][: min(self.grid.shape[i], self.max_bond)]
            self.i[i] = np.column_stack(current_index, self.i[i - 1][: len(current_index)])

    # TODO Move the bond initialization here
    def _create_bond_dimensions(self):
        pass

    # TODO Loops are probably not the best idea here, but we can optimize them later
    def compute_single_site_tensor(self, site):
        tensor = np.ndarray((self.bonds[site - 1], len(self.grid[site]), self.bonds[site]), dtype=self.f_type)

        for left in self.i[site - 1]:
            for right in self.j[site]:
                for i in self.grid[site]:
                    point = np.concatenate((left, i, right))
                    tensor[left, i, right] = self.func(point)

        return tensor

    # TODO Loops are probably not the best idea here, but we can optimize them later
    def compute_superblock_tensor(self, site):
        tensor = np.ndarray(
            (self.bonds[site - 1], len(self.grid[site]), len(self.grid[site + 1]), self.bonds[site + 1]),
            dtype=self.f_type,
        )
        for left in self.i[site - 1]:
            for right in self.j[site + 1]:
                for i in self.grid[site]:
                    for j in self.grid[site + 1]:
                        point = np.concatenate((left, i, j, right))
                        tensor[left, i, j, right] = self.func(point)

        return tensor

    def _crate_initial_arrays(self):
        self.p = np.ndarray(self.num_variables + 1, dtype=np.ndarray)
        self.p[-1] = np.ones((1))
        self.r = np.ones((1))

        self._create_bond_dimensions()
        self._create_initial_index_sets()

        self.b = np.ndarray(self.num_variables, dtype=np.ndarray)

        for i in range(self.num_variables):
            self.b[i] = self.compute_single_site_tensor(i)

    #   @nb.njit() -> Since we are not using ncon in this part, we can use numba
    def initial_sweep_step(self, pos: int):

        # COPY PASTE FROM SAVOSTYANOV, SHOULD WORK -> Really ugly, though
        rk_1 = self.b[pos].shape[0]
        nk = self.b[pos].shape[1]
        rk = self.b[pos].shape[2]

        self.b[pos] = ncon([self.b[pos], self.r], [[-1, -2, 1], [1, -3]])
        c = np.reshape(self.b[pos], (rk_1, nk * rk))
        q_t, r_t = np.linalg.qr(c)
        q = q_t.T, self.r = r_t.T
        q = np.matmul(q, self.p[pos])
        q = np.reshape(q, (rk_1, nk * rk))
        j_k = maxvol(q, complex=True, tol=self.tol, max_iter=1000)

        # Does p maintain the (1,1) shape of the first and last elements?
        self.p[pos - 1] = q[:, j_k]

        # TODO Obtain the index values from the column index j_k
        return j_k

    def initial_sweep(self):
        for site in range(self.num_variables - 1, -1, -1):
            j_k = self.initial_sweep_step(site)
            if site == self.num_variables - 1:
                self.j[site] = np.array([j_k])
            else:
                self.j[site] = np.array(
                    [j_k, self.j[site + 1]]
                )  # TODO: UPDATE I AND J ARRAYS TO BE WHAT THEY ARE SUPPOSED TO BE

    def compute_superblock(self, k):
        # TODO Once the function structure is defined, we need to define how do we compute its values with indices reshaped
        c = np.ndarray((self.i[k - 1].shape, self.grid[k].shape, self.grid[k + 1].shape, self.j[k + 1].shape))

    def right_left_update(self):
        pass

    def left_right_update(self):
        pass

    def left_right_sweep(self):
        pass

    def right_left_sweep(self):
        pass

    def run(self):
        pass

import numpy as np
import numba as nb
from ncon import ncon
from .maxvol import maxvol


class ttrc:
    def __init__(self, func, num_variables, grid, tol, sweeps, initial_bond_guess):
        self.func = func
        self.num_variables = num_variables
        self.grid = grid
        self.tol = tol
        self.sweeps = sweeps
        self.max_bond = initial_bond_guess

    def _crate_initial_arrays(self):
        # TODO For b we could select a bunch of random indices and compute it from func -> Better kickstart
        # TODO Define how we want to intorduce the function, grid and number of variables into the initialization

        self.b = np.ndarray(self.num_variables, dtype=np.ndarray)
        self.b[0] = np.ones((1, len(self.grid[0]), self.max_bond))
        self.b[-1] = np.ones((self.max_bond, len(self.grid[-1]), 1))
        for i in range(1, self.num_variables - 1):
            self.b[i] = np.ones((self.max_bond, len(self.grid[i]), self.max_bond))

        self.p = np.ndarray(self.num_variables + 1, dtype=np.ndarray)
        self.p[0] = np.ones((1))
        self.p[-1] = np.ones((1))
        self.r = np.ones((1))

        # Not really sure what the j and i arrays are supposed to be. They are the caligraphic J and I in the paper

        self.j = np.ndarray(self.num_variables, dtype=np.ndarray)
        self.j[-1] = np.array([1])
        self.i = np.ndarray(self.num_variables, dtype=np.ndarray)
        self.j[0] = np.array([1])

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
        return j_k

    def initial_sweep(self):
        # how do we interpret the nestedness in terms of the columns/rows instead of the indices used in the paper?

        for site in range(self.num_variables - 1, -1, -1):
            j_k = self.initial_sweep_step(site)
            if site == self.num_variables - 1:
                self.j[site] = np.array([j_k])
            else:
                self.j[site] = np.array([j_k, self.j[site + 1]])  # TODO: Check if this is the nestedness we need

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

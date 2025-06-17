from copy import deepcopy
import numpy as np
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator, gmres
from ncon import ncon
from .operators import Prolongation
from ..tns.mps_operations import OrthoOps
from ..tns.mps import create_random_mps

class ALS:
    """
    DMRG style ALS class. It contains all the machinery to obtain a a compressed function psi that satisfies operator * psi = func

    Args:
        - func (np.ndarray): The QTT representation of the target function.
        - operator (np.ndarray): The MPO representation of the operator.
        - initial_bond_guess (int): The initial maximum bond dimension guess for the MPS.
        - max_bond_dim (int): The maximum bond dimension alloed in the two site DMRG style procedure
        - tol (float): Truncation tolerance in the svd steps.
        - sweeps (int): The number of sweeps to perform. By sweep it is meant a full left to right and then right to
            left sequence of updates along the MPS.
    """

    def __init__(
        self,
        func: np.ndarray,
        operator: np.ndarray,
        initial_bonds_guess:int,
        max_bond_dim:int,
        tol: float,
        sweeps: int,
        init_mps: np.ndarray = None
    ):
        copied_list = [deepcopy(tens) for tens in func]

        # 1. Create an empty array with the correct size and dtype=object
        self.func = np.empty(len(copied_list), dtype=object)

        # 2. Fill the empty array with the list of arrays
        self.func[:] = copied_list

        if init_mps is None:
            self.opt_mps = create_random_mps(len(func), initial_bonds_guess, complex_entries=False)
        
        else:
            self.opt_mps = init_mps

        self.opt_mps, _ = OrthoOps.to_right_orthogonal(self.opt_mps, dummy_ends=False)

        self.mpo = operator
        self.L = len(self.func)

        self._check_mpo_mps_compatibility()

        self.bonds = [tensor.shape[-1] for tensor in self.func[:self.L - 1]]
        self.max_chi = max_bond_dim
        self.tol = tol
        self.sweeps = sweeps
        self.cost = []
        self.truncation_error = []

        self._initialize_envs()

    def _check_mpo_mps_compatibility(self):
        if len(self.func) != len(self.mpo):
            raise ValueError(f"Given function MPS and MPO do not share the same length: len(func) = {len(self.func)} != len(mpo) = {len(self.mpo)}")
        
        if self.func[0].shape[0] != self.mpo[0].shape[0]:
            raise ValueError("Func and MPO physical indices do not match at site 0")
        
        for site in range(1, self.L):
            if self.func[site].shape[1] != self.mpo[site].shape[1]:
                raise ValueError(f"Func and MPO physical indices do not match at site {site}")
            
    def _initialize_envs(self):
        """
        Initializes the left and right environment blocks for the energy expectation value. Only the right blocks are
        computed here, as the left blocks are computed on the fly during the first left to right sweep. The convention
        used for the blocks is the following:

            Left blocks:

            psi      -->-->-->--...     |——————|-->-->--...    |——————|-->--...
                    |  |  |  |       =  |L_0   |  |  |      =  |L_1   |          = ...
            MPO     |  0--0--0--...     |——————|--0--0--...    |——————|--0--...
                    |  |  |  |          |      |  |  |         |      |
            psi_0    -->-->-->--...     |——————|-->-->--...    |——————|-->--...

            Right blocks:

            psi     ...--<--<--<--      --<--<--|——————|     ...--<--|——————|
                         |  |  |  |  =    |  |  |R_n-1 |  =       |  |R_n-2 |
            MPO     ...--0--0--0  |     --0--0--|——————|     ...--0--|——————|
                         |  |  |  |       |  |  |      |          |  |      |
            psi_0   ...--<--<--<--      --<--<--|——————|     ...--<--|——————|
        """

        self.l = np.ndarray(self.L, dtype=object)
        self.r = np.ndarray(self.L, dtype=object)

        self.lfg = np.ndarray(self.L, dtype=object)
        self.rfg = np.ndarray(self.L, dtype=object)

        for site in range(self.L - 1, 0, -1):
            self._right_envs_update(site)

    def _left_envs_update(self, site: int):
        """Creates/updates the left environment blocks that participate in the energy expectation value from the current
        MPS and the MPO.

        Args:
            site (int): Site of the MPS where the left environment block is to be created/updated.
        """
        if site == 0:
            self.l[site] = ncon(
                [self.opt_mps[site], self.mpo[site], np.conj(self.opt_mps[site])],
                [[1, -1], [1, 2, -2], [2, -3]]
            )
            
            self.lfg[site] = ncon(
                [self.func[site], np.conj(self.opt_mps[site])],
                [[1, -1], [1, -2]]
            )

            # self.l[site] = ncon(
            #     [self.coarse_mps[site], self.mpo],
            #     [[1, -1], [1, -3, -2]],
            # )

            # self.l[site] = ncon(
            #     [self.l[site], np.conj(self.fine_mps[site])],
            #     [[-1, -2, 1], [1, -3]],
            # )

        else:
            self.l[site] = ncon(
                [self.l[site - 1], self.opt_mps[site], self.mpo[site], np.conj(self.opt_mps[site])],
                [[1, 3, 5], [1, 2, -1], [3, 2, 4, -2], [5, 4, -3]]
            )

            self.lfg[site] = ncon(
                [self.lfg[site - 1], self.func[site], np.conj(self.opt_mps[site])],
                [[1, 2], [1, 3, -1], [2, 3, -2]]
            )

            # self.l[site] = ncon(
            #     [self.l[site - 1], self.coarse_mps[site]],
            #     [[1, -3, -4], [1, -2, -1]],
            # )

            # self.l[site] = ncon(
            #     [self.l[site], self.mpo[site]],
            #     [[-1, 1, 2, -4], [2, 1, -3, -2]],
            # )

            # self.l[site] = ncon(
            #     [self.l[site], np.conj(self.fine_mps[site])],
            #     [[-1, -2, 1, 2], [2, 1, -3]],
            # )

    def _right_envs_update(self, site: int):
        """Creates/updates the right environment blocks that participate in the energy expectation value from the
        current MPS and the MPO.

        Args:
            site (int): Site of the MPS where the right environment block is to be created/updated.
        """
        if site == self.L - 1:
            self.r[site] = ncon(
                [self.opt_mps[site], self.mpo[site], np.conj(self.opt_mps[site])],
                [[-1, 1], [-2, 1, 2], [-3, 2]],
            )

            self.rfg[site] = ncon(
                [self.func[site], np.conj(self.opt_mps[site])],
                [[-1, 1], [-2, 1]]
            )

        else:
            self.r[site] = ncon(
                [self.opt_mps[site], self.mpo[site], np.conj(self.opt_mps[site]), self.r[site + 1]],
                [[-1, 2, 1], [-2, 2, 4, 3], [-3, 4, 5], [1, 3, 5]]
            )

            self.rfg[site] = ncon(
                [self.func[site], np.conj(self.opt_mps[site]), self.rfg[site + 1]],
                [[-1, 2, 1], [-2, 2, 3], [1, 3]]
            )

            # self.r[site] = ncon(
            #     [self.coarse_mps[site], self.r[site + 1]],
            #     [[-1, -2, 1], [1, -3, -4]],
            # )

            # self.r[site] = ncon(
            #     [self.mpo[site], self.r[site]],
            #     [[-2, 1, -3, 2], [-1, 1, 2, -4]],
            # )

            # self.r[site] = ncon(
            #     [np.conj(self.fine_mps[site]), self.r[site]],
            #     [[-3, 1, 2], [-1, -2, 1, 2]],
            # )

    def _leftmost_linearsys_sol(self):
        shape = (self.opt_mps[0].shape[0], self.opt_mps[1].shape[1], self.opt_mps[1].shape[2])
        vec_length = np.prod(shape)

        initial_guess = ncon(
            [self.opt_mps[0], self.opt_mps[1]],
            [[-1, 1], [1, -2, -3]]
        ).reshape((vec_length, ))

        # initial_guess = np.random.randn(vec_length)

        b = ncon(
            [self.func[0], self.func[1], self.rfg[2]], 
            [[-1, 1], [1, -2, 2], [2, -3]]
        )

        if b.shape != shape:
            raise IndexError("Ax and B shapes are not qual in the leftmost linearsys")
        
        else:
            b = b.reshape((vec_length, ))

        def apply_mpo(vec: np.ndarray):
            tens = vec.reshape(shape)

            output = ncon(
                [tens, self.mpo[0], self.mpo[1], self.r[2]],
                [[1, 2, 3], [1, -1, 4], [4, 2, -2, 5], [3, 5, -3]]
            )

            return output.reshape((vec_length, ))
        

        lin_operator = LinearOperator(shape=(vec_length, vec_length), matvec = apply_mpo)

        sol, exitcode = gmres(A = lin_operator, b = b, x0 = initial_guess)

        if exitcode != 0:
            print("Convergence not achieved in leftmost tensor")

        return sol.reshape(shape)
            
    def _rightmost_linearsys_sol(self):
        shape = (self.opt_mps[-2].shape[0], self.opt_mps[-2].shape[1], self.opt_mps[-1].shape[1])
        vec_length = np.prod(shape)

        initial_guess = ncon(
            [self.opt_mps[-2], self.opt_mps[-1]],
            [[-1, -2, 1], [1, -3]]
        ).reshape((vec_length, ))

        # initial_guess = np.random.randn(vec_length)

        b = ncon(
            [self.lfg[-3], self.func[-2], self.func[-1]], 
            [[1, -1], [1, -2, 2], [2, -3]]
        )

        if b.shape != shape:
            raise IndexError("Ax and B shapes are not qual in the rightmost linearsys")
        
        else:
            b = b.reshape((vec_length, ))

        def apply_mpo(vec: np.ndarray):
            tens = vec.reshape(shape)

            output = ncon(
                [self.l[-3], tens, self.mpo[-2], self.mpo[-1]],
                [[1, 4, -1], [1, 2, 3], [4, 2, -2, 5], [5, 3, -3]]   
            )

            return output.reshape((vec_length, ))

        lin_operator = LinearOperator(shape=(vec_length, vec_length), matvec = apply_mpo)

        sol, exitcode = gmres(A = lin_operator, b = b, x0 = initial_guess)

        if exitcode != 0:
            print("Convergence not achieved in rightmost tensor")

        return sol.reshape(shape)
    
    def _inner_linearsys_sol(self, site):
        shape = (self.opt_mps[site].shape[0], self.opt_mps[site].shape[1], self.opt_mps[site + 1].shape[1], self.opt_mps[site + 1].shape[2])
        vec_length = np.prod(shape)

        # initial_guess = np.random.randn(vec_length)

        initial_guess = ncon(
            [self.opt_mps[site], self.opt_mps[site + 1]],
            [[-1, -2, 1], [1, -3, -4]]
        ).reshape((vec_length, ))

        b = ncon(
            [self.lfg[site - 1], self.func[site], self.func[site + 1], self.rfg[site + 2]], 
            [[1, -1], [1, -2, 2], [2, -3, 3], [3, -4]]
        )

        if b.shape != shape:
            raise IndexError(f"Ax and B shapes are not qual in the linearsys at site {site}")
        
        else:
            b = b.reshape((vec_length, ))

        def apply_mpo(vec: np.ndarray):
            tens = vec.reshape(shape)

            output = ncon(
                [self.l[site - 1], tens, self.mpo[site], self.mpo[site + 1], self.r[site + 2]],
                [[1, 5, -1], [1, 2, 3, 4], [5, 2, -2, 6], [6, 3, -3, 7], [4, 7, -4]]   
            )

            return output.reshape((vec_length, ))
        

        lin_operator = LinearOperator(shape=(vec_length, vec_length), matvec = apply_mpo)

        sol, exitcode = gmres(A = lin_operator, b = b, x0 = initial_guess)

        if exitcode != 0:
            print(f"Convergence not achieved site {site}")

        return sol.reshape(shape)
    
    def _leftmost_update(self, left2right:bool = True):
        new_tensor = self._leftmost_linearsys_sol()

        leg_sizes = new_tensor.shape

        new_tensor = np.reshape(new_tensor, (leg_sizes[0], leg_sizes[1] * leg_sizes[2]))
        u, s, v = la.svd(new_tensor, full_matrices=False)

        stemp_cumsum = np.cumsum(s)
        chitemp = int(min(np.argmax(stemp_cumsum >= (1 - self.tol) * stemp_cumsum[-1]) + 1, self.max_chi))
        left_tensor = np.reshape(u[:, :chitemp], (leg_sizes[0], chitemp))
        right_tensor = np.reshape(v[:chitemp, :], (chitemp, leg_sizes[1], leg_sizes[2]))
        s_renorm = np.diag(s[:chitemp])
        self.truncation_error.append(np.sum(s[chitemp:]) / np.sum(s))

        if left2right:
            self.opt_mps[0] = left_tensor
            self.opt_mps[1] = ncon([s_renorm, right_tensor], [[-1, 1], [1, -2, -3]])
        else:
            self.opt_mps[0] = ncon([left_tensor, s_renorm], [[-1, 1], [1, -2]])
            self.opt_mps[1] = right_tensor

        cost1 = ncon(
            [self.opt_mps[0], self.opt_mps[1], self.mpo[0], self.mpo[1], np.conj(self.opt_mps[0]), np.conj(self.opt_mps[1]), self.r[2]],
            [[1, 2], [2, 3, 4], [1, 5, 6], [6, 3, 8, 9], [5, 7], [7, 8, 10], [4, 9, 10]]
        )

        cost2 = ncon(
            [self.func[0], self.func[1], np.conj(self.opt_mps[0]), np.conj(self.opt_mps[1]), self.rfg[2]],
            [[1, 2], [2, 4, 5], [1, 3], [3, 4, 6], [5, 6]]
        )

        self.cost.append(cost1 - cost2)

    def _rightmost_update(self, left2right:bool = True):
        new_tensor = self._rightmost_linearsys_sol()

        leg_sizes = new_tensor.shape

        new_tensor = np.reshape(new_tensor, (leg_sizes[0] * leg_sizes[1], leg_sizes[2]))
        u, s, v = la.svd(new_tensor, full_matrices=False)

        stemp_cumsum = np.cumsum(s)
        chitemp = int(min(np.argmax(stemp_cumsum >= (1 - self.tol) * stemp_cumsum[-1]) + 1, self.max_chi))
        left_tensor = np.reshape(u[:, :chitemp], (leg_sizes[0], leg_sizes[1], chitemp))
        right_tensor = np.reshape(v[:chitemp, :], (chitemp, leg_sizes[2]))
        s_renorm = np.diag(s[:chitemp])
        self.truncation_error.append(np.sum(s[chitemp:]) / np.sum(s))


        if left2right:
            self.opt_mps[self.L - 2] = left_tensor
            self.opt_mps[self.L - 1] = ncon([s_renorm, right_tensor], [[-1, 1], [1, -2]])
        else:
            self.opt_mps[self.L - 2] = ncon([left_tensor, s_renorm], [[-1, -2, 1], [1, -3]])
            self.opt_mps[self.L - 1] = right_tensor

        new_tensor = np.reshape(new_tensor, (leg_sizes[0], leg_sizes[1], leg_sizes[2]))

        cost1 = ncon(
            [self.l[-3], self.opt_mps[-2], self.opt_mps[-1], self.mpo[-2], self.mpo[-1], np.conj(self.opt_mps[-2]), np.conj(self.opt_mps[-1])],
            [[1, 4, 8], [1, 2, 3], [3, 7], [4, 2, 5, 6], [6, 7, 10], [8, 5, 9], [9, 10]]
        )

        cost2 = ncon(
            [self.lfg[-3], self.func[-2], self.func[-1], np.conj(self.opt_mps[-2]), np.conj(self.opt_mps[-1])],
            [[1, 5], [1, 2, 3], [3, 7], [5, 2, 6], [6, 7]]
        )

        self.cost.append(cost1 - cost2)

    def _inner_update(self, site: int, left2right: bool = True):
        new_tensor = self._inner_linearsys_sol(site)
        leg_sizes = new_tensor.shape

        new_tensor = np.reshape(new_tensor, (leg_sizes[0] * leg_sizes[1], leg_sizes[2] * leg_sizes[3]))
        u, s, v = la.svd(new_tensor, full_matrices=False)

        stemp_cumsum = np.cumsum(s)
        chitemp = int(min(np.argmax(stemp_cumsum >= (1 - self.tol) * stemp_cumsum[-1]) + 1, self.max_chi))
        left_tensor = np.reshape(u[:, :chitemp], (leg_sizes[0], leg_sizes[1], chitemp))
        right_tensor = np.reshape(v[:chitemp, :], (chitemp, leg_sizes[2], leg_sizes[3]))
        s_renorm = np.diag(s[:chitemp])
        self.truncation_error.append(np.sum(s[chitemp:]) / np.sum(s))

        if left2right:
            self.opt_mps[site] = left_tensor
            self.opt_mps[site + 1] = ncon([s_renorm, right_tensor], [[-1, 1], [1, -2, -3]])
        else:
            self.opt_mps[site] = ncon([left_tensor, s_renorm], [[-1, -2, 1], [1, -3]])
            self.opt_mps[site + 1] = right_tensor


        cost1 = ncon(
            [self.l[site - 1], self.opt_mps[site], self.opt_mps[site + 1], self.mpo[site], self.mpo[site + 1], np.conj(self.opt_mps[site]), np.conj(self.opt_mps[site +1]), self.r[site + 2]],
            [[1, 6, 9], [1, 2, 3], [3, 4, 5], [6, 2, 10, 7], [7, 4, 12, 8], [9, 10, 11], [11, 12, 13], [5, 8, 13]]
        )

        cost2 = ncon(
            [self.lfg[site - 1], self.func[site], self.func[site + 1], np.conj(self.opt_mps[site ]), np.conj(self.opt_mps[site + 1]), self.rfg[site + 2]],
            [[1, 6], [1, 2, 3], [3, 4, 5], [6, 2, 7], [7, 4, 8], [5, 8]]
        )

        self.cost.append(cost1 - cost2)

    def _left_to_right_sweep(self):
        self._leftmost_update(left2right=True)
        self._left_envs_update(0)

        for site in range(1, self.L - 2):
            self._inner_update(site, left2right=True)
            self._left_envs_update(site)

    def _right_to_left_sweep(self):
        self._rightmost_update(left2right=False)
        self._right_envs_update(self.L - 1)

        for site in range(self.L - 3, 0, -1):
            self._inner_update(site, left2right=False)
            self._right_envs_update(site + 1)

    def optimize(self):
        for _ in range(self.sweeps):
            self._left_to_right_sweep()
            self._right_to_left_sweep()

        return self.opt_mps, self.cost, self.truncation_error
    

class ALS2:
    """
    DMRG style ALS class. It contains all the machinery to obtain a a compressed function psi that satisfies operator * psi = func

    Args:
        - func (np.ndarray): The QTT representation of the target function.
        - operator (np.ndarray): The MPO representation of the operator.
        - initial_bond_guess (int): The initial maximum bond dimension guess for the MPS.
        - max_bond_dim (int): The maximum bond dimension alloed in the two site DMRG style procedure
        - tol (float): Truncation tolerance in the svd steps.
        - sweeps (int): The number of sweeps to perform. By sweep it is meant a full left to right and then right to
            left sequence of updates along the MPS.
    """

    def __init__(
        self,
        func: np.ndarray,
        operator: np.ndarray,
        initial_bonds_guess:int,
        max_bond_dim:int,
        tol: float,
        sweeps: int,
    ):
        self.func = deepcopy(func)

        self.opt_mps = create_random_mps(len(func), initial_bonds_guess, complex_entries=False)
        self.opt_mps, _ = OrthoOps.to_right_orthogonal(self.opt_mps, dummy_ends=False)

        self.mpo = operator
        self.L = len(self.func)

        self._check_mpo_mps_compatibility()

        self.bonds = [tensor.shape[-1] for tensor in self.func[:self.L - 1]]
        self.max_chi = max_bond_dim
        self.tol = tol
        self.sweeps = sweeps
        self.cost = []
        self.truncation_error = []

        self._initialize_envs()

    def _check_mpo_mps_compatibility(self):
        if len(self.func) != len(self.mpo):
            raise ValueError(f"Given function MPS and MPO do not share the same length: len(func) = {len(self.func)} != len(mpo) = {len(self.mpo)}")
        
        if self.func[0].shape[0] != self.mpo[0].shape[0]:
            raise ValueError("Func and MPO physical indices do not match at site 0")
        
        for site in range(1, self.L):
            if self.func[site].shape[1] != self.mpo[site].shape[1]:
                raise ValueError(f"Func and MPO physical indices do not match at site {site}")
            
    def _initialize_envs(self):
        """
        Initializes the left and right environment blocks for the energy expectation value. Only the right blocks are
        computed here, as the left blocks are computed on the fly during the first left to right sweep. The convention
        used for the blocks is the following:

            Left blocks:

            psi      -->-->-->--...     |——————|-->-->--...    |——————|-->--...
                    |  |  |  |       =  |L_0   |  |  |      =  |L_1   |          = ...
            MPO     |  0--0--0--...     |——————|--0--0--...    |——————|--0--...
                    |  |  |  |          |      |  |  |         |      |
            psi_0    -->-->-->--...     |——————|-->-->--...    |——————|-->--...

            Right blocks:

            psi     ...--<--<--<--      --<--<--|——————|     ...--<--|——————|
                         |  |  |  |  =    |  |  |R_n-1 |  =       |  |R_n-2 |
            MPO     ...--0--0--0  |     --0--0--|——————|     ...--0--|——————|
                         |  |  |  |       |  |  |      |          |  |      |
            psi_0   ...--<--<--<--      --<--<--|——————|     ...--<--|——————|
        """

        self.l = np.ndarray(self.L, dtype=object)
        self.r = np.ndarray(self.L, dtype=object)

        self.lfg = np.ndarray(self.L, dtype=object)
        self.rfg = np.ndarray(self.L, dtype=object)

        for site in range(self.L - 1, 0, -1):
            self._right_envs_update(site)

    def _left_envs_update(self, site: int):
        """Creates/updates the left environment blocks that participate in the energy expectation value from the current
        MPS and the MPO.

        Args:
            site (int): Site of the MPS where the left environment block is to be created/updated.
        """
        if site == 0:
            self.l[site] = ncon(
                [self.opt_mps[site], self.mpo[site], np.conj(self.opt_mps[site])],
                [[1, -1], [1, 2, -2], [2, -3]]
            )
            
            self.lfg[site] = ncon(
                [self.func[site], np.conj(self.opt_mps[site])],
                [[1, -1], [1, -2]]
            )

            # self.l[site] = ncon(
            #     [self.coarse_mps[site], self.mpo],
            #     [[1, -1], [1, -3, -2]],
            # )

            # self.l[site] = ncon(
            #     [self.l[site], np.conj(self.fine_mps[site])],
            #     [[-1, -2, 1], [1, -3]],
            # )

        else:
            self.l[site] = ncon(
                [self.l[site - 1], self.opt_mps[site], self.mpo[site], np.conj(self.opt_mps[site])],
                [[1, 3, 5], [1, 2, -1], [3, 2, 4, -2], [5, 4, -3]]
            )

            self.lfg[site] = ncon(
                [self.lfg[site - 1], self.func[site], np.conj(self.opt_mps[site])],
                [[1, 2], [1, 3, -1], [2, 3, -2]]
            )

            # self.l[site] = ncon(
            #     [self.l[site - 1], self.coarse_mps[site]],
            #     [[1, -3, -4], [1, -2, -1]],
            # )

            # self.l[site] = ncon(
            #     [self.l[site], self.mpo[site]],
            #     [[-1, 1, 2, -4], [2, 1, -3, -2]],
            # )

            # self.l[site] = ncon(
            #     [self.l[site], np.conj(self.fine_mps[site])],
            #     [[-1, -2, 1, 2], [2, 1, -3]],
            # )

    def _right_envs_update(self, site: int):
        """Creates/updates the right environment blocks that participate in the energy expectation value from the
        current MPS and the MPO.

        Args:
            site (int): Site of the MPS where the right environment block is to be created/updated.
        """
        if site == self.L - 1:
            self.r[site] = ncon(
                [self.opt_mps[site], self.mpo[site], np.conj(self.opt_mps[site])],
                [[-1, 1], [-2, 1, 2], [-3, 2]],
            )

            self.rfg[site] = ncon(
                [self.func[site], np.conj(self.opt_mps[site])],
                [[-1, 1], [-2, 1]]
            )

        else:
            self.r[site] = ncon(
                [self.opt_mps[site], self.mpo[site], np.conj(self.opt_mps[site]), self.r[site + 1]],
                [[-1, 2, 1], [-2, 2, 4, 3], [-3, 4, 5], [1, 3, 5]]
            )

            self.rfg[site] = ncon(
                [self.func[site], np.conj(self.opt_mps[site]), self.rfg[site + 1]],
                [[-1, 2, 1], [-2, 2, 3], [1, 3]]
            )

            # self.r[site] = ncon(
            #     [self.coarse_mps[site], self.r[site + 1]],
            #     [[-1, -2, 1], [1, -3, -4]],
            # )

            # self.r[site] = ncon(
            #     [self.mpo[site], self.r[site]],
            #     [[-2, 1, -3, 2], [-1, 1, 2, -4]],
            # )

            # self.r[site] = ncon(
            #     [np.conj(self.fine_mps[site]), self.r[site]],
            #     [[-3, 1, 2], [-1, -2, 1, 2]],
            # )

    def _leftmost_linearsys_sol(self):
        shape = (self.opt_mps[0].shape[0], self.opt_mps[1].shape[1], self.opt_mps[1].shape[2])
        vec_length = np.prod(shape)

        initial_guess = ncon(
            [self.opt_mps[0], self.opt_mps[1]],
            [[-1, 1], [1, -2, -3]]
        ).reshape((vec_length, ))

        # initial_guess = np.random.randn(vec_length)

        b = 2 * ncon(
            [self.func[0], self.func[1], self.rfg[2]], 
            [[-1, 1], [1, -2, 2], [2, -3]]
        )

        if b.shape != shape:
            raise IndexError("Ax and B shapes are not qual in the leftmost linearsys")
        
        else:
            b = b.reshape((vec_length, ))

        def apply_mpo(vec: np.ndarray):
            tens = vec.reshape(shape)

            output = ncon(
                [tens, self.mpo[0], self.mpo[1], self.r[2]],
                [[1, 2, 3], [1, -1, 4], [4, 2, -2, 5], [3, 5, -3]]
            )

            return output.reshape((vec_length, ))
        

        lin_operator = LinearOperator(shape=(vec_length, vec_length), matvec = apply_mpo)

        sol, exitcode = gmres(A = lin_operator, b = b, x0 = initial_guess)

        if exitcode != 0:
            print("Convergence not achieved in leftmost tensor")

        return sol.reshape(shape)
            
    def _rightmost_linearsys_sol(self):
        shape = (self.opt_mps[-2].shape[0], self.opt_mps[-2].shape[1], self.opt_mps[-1].shape[1])
        vec_length = np.prod(shape)

        initial_guess = ncon(
            [self.opt_mps[-2], self.opt_mps[-1]],
            [[-1, -2, 1], [1, -3]]
        ).reshape((vec_length, ))

        # initial_guess = np.random.randn(vec_length)

        b = 2 * ncon(
            [self.lfg[-3], self.func[-2], self.func[-1]], 
            [[1, -1], [1, -2, 2], [2, -3]]
        )

        if b.shape != shape:
            raise IndexError("Ax and B shapes are not qual in the leftmost linearsys")
        
        else:
            b = b.reshape((vec_length, ))

        def apply_mpo(vec: np.ndarray):
            tens = vec.reshape(shape)

            output = ncon(
                [self.l[-3], tens, self.mpo[-2], self.mpo[-1]],
                [[1, 4, -1], [1, 2, 3], [4, 2, -2, 5], [5, 3, -3]]   
            )

            return output.reshape((vec_length, ))

        lin_operator = LinearOperator(shape=(vec_length, vec_length), matvec = apply_mpo)

        sol, exitcode = gmres(A = lin_operator, b = b, x0 = initial_guess)

        if exitcode != 0:
            print("Convergence not achieved in rightmost tensor")

        return sol.reshape(shape)
    
    def _inner_linearsys_sol(self, site):
        shape = (self.opt_mps[site].shape[0], self.opt_mps[site].shape[1], self.opt_mps[site + 1].shape[1], self.opt_mps[site + 1].shape[2])
        vec_length = np.prod(shape)

        # initial_guess = np.random.randn(vec_length)

        initial_guess = ncon(
            [self.opt_mps[site], self.opt_mps[site + 1]],
            [[-1, -2, 1], [1, -3, -4]]
        ).reshape((vec_length, ))

        b = 2 * ncon(
            [self.lfg[site - 1], self.func[site], self.func[site + 1], self.rfg[site + 2]], 
            [[1, -1], [1, -2, 2], [2, -3, 3], [3, -4]]
        )

        if b.shape != shape:
            raise IndexError("Ax and B shapes are not qual in the leftmost linearsys")
        
        else:
            b = b.reshape((vec_length, ))

        def apply_mpo(vec: np.ndarray):
            tens = vec.reshape(shape)

            output = ncon(
                [self.l[site - 1], tens, self.mpo[site], self.mpo[site + 1], self.r[site + 2]],
                [[1, 5, -1], [1, 2, 3, 4], [5, 2, -2, 6], [6, 3, -3, 7], [4, 7, -4]]   
            )

            return output.reshape((vec_length, ))
        

        lin_operator = LinearOperator(shape=(vec_length, vec_length), matvec = apply_mpo)

        sol, exitcode = gmres(A = lin_operator, b = b, x0 = initial_guess)

        if exitcode != 0:
            print(f"Convergence not achieved in tensor at site {site}")

        return sol.reshape(shape)
    
    def _leftmost_update(self, left2right:bool = True):
        new_tensor = self._leftmost_linearsys_sol()

        leg_sizes = new_tensor.shape

        new_tensor = np.reshape(new_tensor, (leg_sizes[0], leg_sizes[1] * leg_sizes[2]))
        u, s, v = la.svd(new_tensor, full_matrices=False)

        stemp_cumsum = np.cumsum(s)
        chitemp = int(min(np.argmax(stemp_cumsum >= (1 - self.tol) * stemp_cumsum[-1]) + 1, self.max_chi))
        left_tensor = np.reshape(u[:, :chitemp], (leg_sizes[0], chitemp))
        right_tensor = np.reshape(v[:chitemp, :], (chitemp, leg_sizes[1], leg_sizes[2]))
        s_renorm = np.diag(s[:chitemp])
        self.truncation_error.append(np.sum(s[chitemp:]) / np.sum(s))

        if left2right:
            self.opt_mps[0] = left_tensor
            self.opt_mps[1] = ncon([s_renorm, right_tensor], [[-1, 1], [1, -2, -3]])
        else:
            self.opt_mps[0] = ncon([left_tensor, s_renorm], [[-1, 1], [1, -2]])
            self.opt_mps[1] = right_tensor

        cost1 = ncon(
            [self.opt_mps[0], self.opt_mps[1], self.mpo[0], self.mpo[1], np.conj(self.opt_mps[0]), np.conj(self.opt_mps[1]), self.r[2]],
            [[1, 2], [2, 3, 4], [1, 5, 6], [6, 3, 8, 9], [5, 7], [7, 8, 10], [4, 9, 10]]
        )

        cost2 = ncon(
            [self.func[0], self.func[1], np.conj(self.opt_mps[0]), np.conj(self.opt_mps[1]), self.rfg[2]],
            [[1, 2], [2, 4, 5], [1, 3], [3, 4, 6], [5, 6]]
        )

        self.cost.append(cost1 - 2 * cost2)

    def _rightmost_update(self, left2right:bool = True):
        new_tensor = self._rightmost_linearsys_sol()

        leg_sizes = new_tensor.shape

        new_tensor = np.reshape(new_tensor, (leg_sizes[0] * leg_sizes[1], leg_sizes[2]))
        u, s, v = la.svd(new_tensor, full_matrices=False)

        stemp_cumsum = np.cumsum(s)
        chitemp = int(min(np.argmax(stemp_cumsum >= (1 - self.tol) * stemp_cumsum[-1]) + 1, self.max_chi))
        left_tensor = np.reshape(u[:, :chitemp], (leg_sizes[0], leg_sizes[1], chitemp))
        right_tensor = np.reshape(v[:chitemp, :], (chitemp, leg_sizes[2]))
        s_renorm = np.diag(s[:chitemp])
        self.truncation_error.append(np.sum(s[chitemp:]) / np.sum(s))


        if left2right:
            self.opt_mps[self.L - 2] = left_tensor
            self.opt_mps[self.L - 1] = ncon([s_renorm, right_tensor], [[-1, 1], [1, -2]])
        else:
            self.opt_mps[self.L - 2] = ncon([left_tensor, s_renorm], [[-1, -2, 1], [1, -3]])
            self.opt_mps[self.L - 1] = right_tensor

        new_tensor = np.reshape(new_tensor, (leg_sizes[0], leg_sizes[1], leg_sizes[2]))

        cost1 = ncon(
            [self.l[-3], self.opt_mps[-2], self.opt_mps[-1], self.mpo[-2], self.mpo[-1], np.conj(self.opt_mps[-2]), np.conj(self.opt_mps[-1])],
            [[1, 4, 8], [1, 2, 3], [3, 7], [4, 2, 5, 6], [6, 7, 10], [8, 5, 9], [9, 10]]
        )

        cost2 = ncon(
            [self.lfg[-3], self.func[-2], self.func[-1], np.conj(self.opt_mps[-2]), np.conj(self.opt_mps[-1])],
            [[1, 5], [1, 2, 3], [3, 7], [5, 2, 6], [6, 7]]
        )

        self.cost.append(cost1 - 2 * cost2)

    def _inner_update(self, site: int, left2right: bool = True):
        new_tensor = self._inner_linearsys_sol(site)
        leg_sizes = new_tensor.shape

        new_tensor = np.reshape(new_tensor, (leg_sizes[0] * leg_sizes[1], leg_sizes[2] * leg_sizes[3]))
        u, s, v = la.svd(new_tensor, full_matrices=False)

        stemp_cumsum = np.cumsum(s)
        chitemp = int(min(np.argmax(stemp_cumsum >= (1 - self.tol) * stemp_cumsum[-1]) + 1, self.max_chi))
        left_tensor = np.reshape(u[:, :chitemp], (leg_sizes[0], leg_sizes[1], chitemp))
        right_tensor = np.reshape(v[:chitemp, :], (chitemp, leg_sizes[2], leg_sizes[3]))
        s_renorm = np.diag(s[:chitemp])
        self.truncation_error.append(np.sum(s[chitemp:]) / np.sum(s))

        if left2right:
            self.opt_mps[site] = left_tensor
            self.opt_mps[site + 1] = ncon([s_renorm, right_tensor], [[-1, 1], [1, -2, -3]])
        else:
            self.opt_mps[site] = ncon([left_tensor, s_renorm], [[-1, -2, 1], [1, -3]])
            self.opt_mps[site + 1] = right_tensor


        cost1 = ncon(
            [self.l[site - 1], self.opt_mps[site], self.opt_mps[site + 1], self.mpo[site], self.mpo[site + 1], np.conj(self.opt_mps[site]), np.conj(self.opt_mps[site +1]), self.r[site + 2]],
            [[1, 6, 9], [1, 2, 3], [3, 4, 5], [6, 2, 10, 7], [7, 4, 12, 8], [9, 10, 11], [11, 12, 13], [5, 8, 13]]
        )

        cost2 = ncon(
            [self.lfg[site - 1], self.func[site], self.func[site + 1], np.conj(self.opt_mps[site ]), np.conj(self.opt_mps[site + 1]), self.rfg[site + 2]],
            [[1, 6], [1, 2, 3], [3, 4, 5], [6, 2, 7], [7, 4, 8], [5, 8]]
        )

        self.cost.append(cost1 - 2 * cost2)

    def _left_to_right_sweep(self):
        self._leftmost_update(left2right=True)
        self._left_envs_update(0)

        for site in range(1, self.L - 2):
            self._inner_update(site, left2right=True)
            self._left_envs_update(site)

    def _right_to_left_sweep(self):
        self._rightmost_update(left2right=False)
        self._right_envs_update(self.L - 1)

        for site in range(self.L - 3, 0, -1):
            self._inner_update(site, left2right=False)
            self._right_envs_update(site + 1)

    def optimize(self):
        for _ in range(self.sweeps):
            self._left_to_right_sweep()
            self._right_to_left_sweep()

        return self.opt_mps, self.cost, self.truncation_error
    
class ProlongationALS:
    """
    DMRG style ALS-based prolongation class. It contains all the machinery to refine a coarse QTT representation of a function

    Args:
        - coarse_func (np.ndarray): The coarse QTT representation of the target function.
        - initial_bond_guess (int): The initial maximum bond dimension guess for the MPS.
        - max_bond_dim (int): The maximum bond dimension alloed in the two site DMRG style procedure
        - tol (float): Truncation tolerance in the svd steps.
        - sweeps (int): The number of sweeps to perform. By sweep it is meant a full left to right and then right to
            left sequence of updates along the MPS.
    """

    def __init__(
        self,
        coarse_func: np.ndarray,
        initial_bonds_guess:int,
        max_bond_dim:int,
        tol: float,
        sweeps: int,
    ):
        
        self.coarse_mps = deepcopy(coarse_func)

        self.fine_mps = create_random_mps(len(coarse_func) + 1, initial_bonds_guess, complex_entries=True)

        self.fine_mps, _ = OrthoOps.to_right_orthogonal(self.fine_mps, dummy_ends=False)
        self.mpo = Prolongation(len(self.coarse_mps)).build_operator()


        self.L = len(self.coarse_mps)
        self.bonds = [tensor.shape[-1] for tensor in self.fine_mps[:self.L + 1]]
        self.max_chi = max_bond_dim
        self.tol = tol
        self.sweeps = sweeps
        self.overlap = []
        self.truncation_error = []
        self._initialize_envs()

    def _initialize_envs(self):
        """
        Initializes the left and right environment blocks for the energy expectation value. Only the right blocks are
        computed here, as the left blocks are computed on the fly during the first left to right sweep. The convention
        used for the blocks is the following:

            Left blocks:

            psi      -->-->-->--...     |——————|-->-->--...    |——————|-->--...
                    |  |  |  |       =  |L_0   |  |  |      =  |L_1   |          = ...
            MPO     |  0--0--0--...     |——————|--0--0--...    |——————|--0--...
                    |  |  |  |          |      |  |  |         |      |
            psi_0    -->-->-->--...     |——————|-->-->--...    |——————|-->--...

            Right blocks:

            psi     ...--<--<--<--      --<--<--|——————|     ...--<--|——————|
                         |  |  |  |  =    |  |  |R_n-1 |  =       |  |R_n-2 |
            MPO     ...--0--0--0  |     --0--0--|——————|     ...--0--|——————|
                         |  |  |  |       |  |  |      |          |  |      |
            psi_0   ...--<--<--<--      --<--<--|——————|     ...--<--|——————|
        """

        self.l = np.ndarray(self.L + 1, dtype=object)
        self.r = np.ndarray(self.L + 1, dtype=object)

        for site in range(self.L, 0, -1):
            self._right_envs_update(site)

    def _left_envs_update(self, site: int):
        """Creates/updates the left environment blocks that participate in the energy expectation value from the current
        MPS and the MPO.

        Args:
            site (int): Site of the MPS where the left environment block is to be created/updated.
        """
        if site == 0:
            self.l[site] = ncon(
                [self.coarse_mps[site], self.mpo[site], np.conj(self.fine_mps[site])],
                [[1, -1], [1, 2, -2], [2, -3]]
            )

        else:
            self.l[site] = ncon(
                [self.l[site - 1], self.coarse_mps[site], self.mpo[site], np.conj(self.fine_mps[site])],
                [[1, 3, 5], [1, 2, -1], [3, 2, 4, -2], [5, 4, -3]]
            )

    def _right_envs_update(self, site: int):
        """Creates/updates the right environment blocks that participate in the energy expectation value from the
        current MPS and the MPO.

        Args:
            site (int): Site of the MPS where the right environment block is to be created/updated.
        """
        if site == self.L:
            self.r[site] = ncon(
                [self.mpo[site], np.conj(self.fine_mps[site])],
                [[-1, 1], [-2, 1]],
            )

        elif site == self.L - 1:
            self.r[site] = ncon(
                [self.coarse_mps[site], self.mpo[site], np.conj(self.fine_mps[site]), self.r[site + 1]],
                [[-1, 1], [-2, 1, 3, 2], [-3, 3, 4], [2, 4]]
            )

        else:
            self.r[site] = ncon(
                [self.coarse_mps[site], self.mpo[site], np.conj(self.fine_mps[site]), self.r[site + 1]],
                [[-1, 2, 1], [-2, 2, 4, 3], [-3, 4, 5], [1, 3, 5]]
            )

    def _leftmost_update(self, left2right:bool = True):
        new_tensor = ncon(
            [self.coarse_mps[0], self.coarse_mps[1], self.mpo[0], self.mpo[1], self.r[2]],
            [[1, 2], [2, 4, 5], [1, -1, 3], [3, 4, -2, 6], [5, 6, -3]]
        )

        leg_sizes = new_tensor.shape

        new_tensor = np.reshape(new_tensor, (leg_sizes[0], leg_sizes[1] * leg_sizes[2]))
        u, s, v = la.svd(new_tensor, full_matrices=False)

        stemp_cumsum = np.cumsum(s)
        chitemp = int(min(np.argmax(stemp_cumsum >= (1 - self.tol) * stemp_cumsum[-1]) + 1, self.max_chi))
        left_tensor = np.reshape(u[:, :chitemp], (leg_sizes[0], chitemp))
        right_tensor = np.reshape(v[:chitemp, :], (chitemp, leg_sizes[1], leg_sizes[2]))
        # s_renorm = np.diag(s[:chitemp] / la.norm(s[:chitemp]))
        s_renorm = np.diag(s[:chitemp])
        self.truncation_error.append(np.sum(s[chitemp:]))

        if left2right:
            self.fine_mps[0] = left_tensor
            self.fine_mps[1] = ncon([s_renorm, right_tensor], [[-1, 1], [1, -2, -3]])
        else:
            self.fine_mps[0] = ncon([left_tensor, s_renorm], [[-1, 1], [1, -2]])
            self.fine_mps[1] = right_tensor

        new_tensor = np.reshape(new_tensor, (leg_sizes[0], leg_sizes[1], leg_sizes[2]))
        self.overlap.append(
            ncon([new_tensor, np.conj(self.fine_mps[0]), np.conj(self.fine_mps[1])],
                 [[1, 3, 4], [1, 2], [2, 3, 4]])
        )

    def _rightmost_update(self, left2right:bool = True):
        new_tensor = ncon(
            [self.l[self.L - 2], self.coarse_mps[self.L - 1], self.mpo[self.L - 1], self.mpo[self.L]],
            [[1, 2, -1], [1, 3], [2, 3, -2, 4], [4, -3]]
        )
        leg_sizes = new_tensor.shape

        new_tensor = np.reshape(new_tensor, (leg_sizes[0] * leg_sizes[1], leg_sizes[2]))
        u, s, v = la.svd(new_tensor, full_matrices=False)

        stemp_cumsum = np.cumsum(s)
        chitemp = int(min(np.argmax(stemp_cumsum >= (1 - self.tol) * stemp_cumsum[-1]) + 1, self.max_chi))
        left_tensor = np.reshape(u[:, :chitemp], (leg_sizes[0], leg_sizes[1], chitemp))
        right_tensor = np.reshape(v[:chitemp, :], (chitemp, leg_sizes[2]))
        # s_renorm = np.diag(s[:chitemp] / la.norm(s[:chitemp]))
        s_renorm = np.diag(s[:chitemp])
        self.truncation_error.append(np.sum(s[chitemp:]))


        if left2right:
            self.fine_mps[self.L - 1] = left_tensor
            self.fine_mps[self.L] = ncon([s_renorm, right_tensor], [[-1, 1], [1, -2]])
        else:
            self.fine_mps[self.L - 1] = ncon([left_tensor, s_renorm], [[-1, -2, 1], [1, -3]])
            self.fine_mps[self.L] = right_tensor

        new_tensor = np.reshape(new_tensor, (leg_sizes[0], leg_sizes[1], leg_sizes[2]))
        self.overlap.append(
            ncon([new_tensor, np.conj(self.fine_mps[self.L - 1]), np.conj(self.fine_mps[self. L])],
                 [[1, 2, 4], [1, 2, 3], [3, 4]])
        )

    def _second_rightmost_update(self, left2right:bool = True):
        new_tensor = ncon(
            [self.l[self.L - 3], self.coarse_mps[self.L - 2], self.coarse_mps[self.L - 1], self.mpo[self.L - 2], self.mpo[self.L - 1], self.r[self.L]],
            [[1, 2, -1], [1, 3, 4], [4, 6], [2, 3, -2, 5], [5, 6, -3, 7], [7, -4]]
        )
        leg_sizes = new_tensor.shape

        new_tensor = np.reshape(new_tensor, (leg_sizes[0] * leg_sizes[1], leg_sizes[2] * leg_sizes[3]))
        u, s, v = la.svd(new_tensor, full_matrices=False)

        stemp_cumsum = np.cumsum(s)
        chitemp = int(min(np.argmax(stemp_cumsum >= (1 - self.tol) * stemp_cumsum[-1]) + 1, self.max_chi))
        left_tensor = np.reshape(u[:, :chitemp], (leg_sizes[0], leg_sizes[1], chitemp))
        right_tensor = np.reshape(v[:chitemp, :], (chitemp, leg_sizes[2], leg_sizes[3]))
        # s_renorm = np.diag(s[:chitemp] / la.norm(s[:chitemp]))
        s_renorm = np.diag(s[:chitemp])
        self.truncation_error.append(np.sum(s[chitemp:]))

        if left2right:
            self.fine_mps[self.L - 2] = left_tensor
            self.fine_mps[self.L - 1] = ncon([s_renorm, right_tensor], [[-1, 1], [1, -2, -3]])
        else:
            self.fine_mps[self.L - 2] = ncon([left_tensor, s_renorm], [[-1, -2, 1], [1, -3]])
            self.fine_mps[self.L - 1] = right_tensor

        new_tensor = np.reshape(new_tensor, (leg_sizes[0], leg_sizes[1], leg_sizes[2], leg_sizes[3]))
        self.overlap.append(
            ncon([new_tensor, np.conj(self.fine_mps[self.L - 2]), np.conj(self.fine_mps[self. L - 1])],
                 [[1, 2, 4, 5], [1, 2, 3], [3, 4, 5]])
        )

    def _inner_update(self, site: int, left2right: bool = True):
        new_tensor = ncon(
            [self.l[site - 1], self.coarse_mps[site], self.coarse_mps[site + 1], self.mpo[site], self.mpo[site + 1], self.r[site + 2]],
            [[1, 2, -1], [1, 3, 4], [4, 6, 7], [2, 3, -2, 5], [5, 6, -3, 8], [7, 8, -4]]
        )
        leg_sizes = new_tensor.shape

        new_tensor = np.reshape(new_tensor, (leg_sizes[0] * leg_sizes[1], leg_sizes[2] * leg_sizes[3]))
        u, s, v = la.svd(new_tensor, full_matrices=False)

        stemp_cumsum = np.cumsum(s)
        chitemp = int(min(np.argmax(stemp_cumsum >= (1 - self.tol) * stemp_cumsum[-1]) + 1, self.max_chi))
        left_tensor = np.reshape(u[:, :chitemp], (leg_sizes[0], leg_sizes[1], chitemp))
        right_tensor = np.reshape(v[:chitemp, :], (chitemp, leg_sizes[2], leg_sizes[3]))
        # s_renorm = np.diag(s[:chitemp] / la.norm(s[:chitemp]))
        s_renorm = np.diag(s[:chitemp])

        self.truncation_error.append(np.sum(s[chitemp:]))

        if left2right:
            self.fine_mps[site] = left_tensor
            self.fine_mps[site + 1] = ncon([s_renorm, right_tensor], [[-1, 1], [1, -2, -3]])
        else:
            self.fine_mps[site] = ncon([left_tensor, s_renorm], [[-1, -2, 1], [1, -3]])
            self.fine_mps[site + 1] = right_tensor

        new_tensor = np.reshape(new_tensor, (leg_sizes[0], leg_sizes[1], leg_sizes[2], leg_sizes[3]))
        self.overlap.append(
            ncon([new_tensor, np.conj(self.fine_mps[site]), np.conj(self.fine_mps[site + 1])],
                 [[1, 2, 4, 5], [1, 2, 3], [3, 4, 5]])
        )

    def _left_to_right_sweep(self):
        self._leftmost_update(left2right=True)
        self._left_envs_update(0)

        for site in range(1, self.L - 2):
            self._inner_update(site, left2right=True)
            self._left_envs_update(site)

        self._second_rightmost_update(left2right=True)
        self._left_envs_update(self.L - 2)

    def _right_to_left_sweep(self):
        self._rightmost_update(left2right=False)
        self._right_envs_update(self.L)

        self._second_rightmost_update(left2right=False)
        self._right_envs_update(self.L - 1)

        for site in range(self.L - 3, 0, -1):
            self._inner_update(site, left2right=False)
            self._right_envs_update(site + 1)

    def optimize(self):
        for _ in range(self.sweeps):
            self._left_to_right_sweep()
            self._right_to_left_sweep()

        return self.fine_mps, self.overlap, self.truncation_error
        

class TimeEvolveALS:
    pass




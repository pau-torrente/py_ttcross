import sys

sys.path.append("/home/ptbadia/code/tfg/tfg_ttcross")

from abc import ABC, abstractmethod
import numpy as np
from tt_cross.src.utils.maxvol import greedy_pivot_finder, py_maxvol
from types import FunctionType
from ncon import ncon
import warnings
import time
from scipy.linalg import svd
import numba as nb

# Helper function that is compiled using numba outside the class to reduce the execution time of the superblock tensor
# computation, which is the most expensive part of the algorithm.

@nb.njit()
def compute_superblock_tensor(i: np.ndarray, j: np.ndarray, g1: np.ndarray, g2: np.ndarray, site: int):
    """Method that using the index sets stored in the self.i and self.j variables computes the superblock tensor
    A(I_{k-1}, i_k, i_{k+1}, J_{k+1}) used to update the index sets in the tensor train in DMRG-like procedures.

    Args:
        site (int): The site of the left physical leg of the superblock tensor.

    Returns:
        np.ndarray: The output 4-legged tensor corresponding to the superblock tensor.
    """
    tensor = np.empty(
        (len(i), len(g1), len(g2), len(j)),
        dtype=np.float64,
    )

    # Run over all the points in the set (I_{k-1}, i_k, i_{k+1}, J_{k+1})
    for s in range(len(i)):
        left = i[s]
        for k in range(len(j)):
            right = j[k]
            for m in range(len(g1)):
                i_1 = np.array([g1[m]], dtype = np.float64)
                for n in range(len(g2)):
                    i_2 = np.array([g2[n]], dtype = np.float64)

                    # And as in the single-site tensor, we consider if we are at the first site, the last site or in
                    # the bulk to avoid adding the dummy index at the start or end of the self.i and self.j sets,
                    # respectively.

                    if site == 0:
                        point = np.concatenate((i_1, i_2, right))
                    elif site == 2:
                        point = np.concatenate((left, i_1, i_2))
                    elif site == 1:
                        point = np.concatenate((left, i_1, i_2, right))

                    tensor[s, m, n, k] = int_function(point)

    return tensor


class tt_interpolator(ABC):
    """
    Class representing a general tensor train interpolator. It only incorporates the methods related to obtaining
    the total indices I_{k-1}⊗i_k and J_{k+1}⊗j_{k+1} for a given site and 3 methods to compute single-site tensors,
    2-site superblock tensors (2-site DMRG style) and the inverse of the cross block for a given site. The class is
    intended to be inherited specific implementations of the interpolator which contain the methods related to
    updating the index sets.

    Some of the most confusing notation used is the following:

    - self.i[k] = {1, i_1^s, i_2^s, ..., i_k^s}, s = 1, ..., r_k

    - self.j[k] = {i_{k+1}^s, i_{k+2}^s, ..., i_N^s, 1}, s = 1, ..., r_k
        (The 1 at the beginning of self.i[k] and end of self.j[k] are dummy indices which are there just for
        simplicity when contracting the tensors)

    - total_indices_left = self.i[k-1]⊗i_k

    - total_indices_right = i_{k+1}⊗self.j[k+1]
        (When doing this outer products, we always take the indices from self.i and self.j first and then the ones
        coming from i_k and i_{k+1} are the ones that get tiled)

    Args:
        - func (FunctionType): The function to be interpolated. It must be a function that takes a numpy array as input
        and returns a float or complex number. For speed purposes, it is highly recommended to pass a function which 
        can be numba jit compiled.
        
        - num_variables (int): The number of variables of the grid on which the function is defined.
        
        - grid (np.ndarray): The grid on which the function is defined. It must be a numpy array of len = num_variables.
            The number of points in each dimension of the grid can be different.
            
        - tol (float): Tolerance for the pivot finding algorithm.
        
        - sweeps (int): Number of sweeps to perform.
        
        - is_f_complex (bool, optional): Whether the function is complex or not. Defaults to False.
        
        - pivot_initialization (str, optional): The way in which the initial pivots are selected. It can be either
        "random" or "first_n". Defaults to "random".

    """

    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        grid: np.ndarray,
        tol: float,
        sweeps: int,
        is_f_complex:bool=False,
        pivot_initialization:str = "random"
    ) -> None:
        self.func = func
        if len(grid) != num_variables:
            raise ValueError("The grid must have the same number of dimensions as the number of variables")
        self.num_variables = num_variables
        self.grid = grid
        self.tol = tol
        self.sweeps = sweeps
        
        if pivot_initialization not in ["random", "first_n"]:
            raise ValueError("Pivot initialization must be either 'random' or 'first_n'.")
        
        self.pivot_init = pivot_initialization

        self.f_type = np.complex128 if is_f_complex else np.float64

        self.bonds = np.ndarray(self.num_variables + 1, dtype=np.ndarray)
        self.bonds[0] = 1
        self.bonds[-1] = 1
        self.super_block_time = 0
        self.func_calls = 0
        
        # If the function can be compiled with numba, we compile it and use the compiled version to compute the
        # superblock tensor. If it cannot be compiled, we use the non-compiled version.
        try:
            # Set the int_function as a global variable such that the external superblock tensor generator can call it
            global int_function
            int_function = nb.jit(nopython = True)(func)
            self.compute_superblock_tensor = self._compute_superblock_tensor_compiled
            print("Function successfully compiled with numba.")
        
        except:
            self.func = func
            self.compute_superblock_tensor = self._compute_superblock_tensor_non_compiled
            print("Function not compiled with numba. Using non-compiled version.")
            
            

    def _obtain_superblock_total_indices(
        self, site: int, compute_index_pos: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list, list]:
        """Obtain the set of all possible indices I_{k-1}⊗i_k and J_{k+1}⊗i_{k+1} for the given site k = site. A part
        from this subsets, it can also return the position of the current best indices I_{k} and J_{k} in these sets of
        total possible subindices.

        Example:
        I_{k-1} = [[1, 2],
                   [3, 4],
                   [5, 6]]

        i_k = [[7],
               [8]]

        total_indices_left = [[1, 2, 7],
                              [1, 2, 8],
                              [3, 4, 7],
                              [3, 4, 8],
                              [5, 6, 7],
                              [5, 6, 8]]

        =============================================================

        J_{k+1} = [[9, 10],
                   [11, 12],
                   [13, 14]]

        j_{k+1} = [[15],
                   [16]]

        total_indices_right = [[15, 9, 10],
                               [16, 9, 10],
                               [15, 11, 12],
                               [16, 11, 12],
                               [15, 13, 14],
                               [16, 13, 14]]


        Args:
            site (int): The left site of the superblock.
            compute_index_pos (bool, optional): Whether to compute also the position of the current best sets of indices
            . Defaults to True.

        Returns:
            tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list, list]: The total indices left and right.
            If compute_index_pos is True, the current best indices positions are also returned.
        """
        # TODO: If np.where is too slow, stack overflow users suggest using hash maps to speed up the process

        total_indices_left = []
        total_indices_right = []

        current_indices_left_pos = []
        current_indices_right_pos = []

        # Since at site k = 0, there is no left index set, we just take all the elements of the grid at site k = 0 and
        if site == 0:
            total_indices_left = np.array([self.grid[site].copy()], dtype=object).T

            # If we are computing the index positions, we just use the np.where function to find the position of the
            # current best pivot in the total indices set.
            if compute_index_pos:
                for current_best_pivot in self.i[site + 1]:
                    current_indices_left_pos.append(
                        np.where(np.all(total_indices_left == current_best_pivot, axis=1))[0][0]
                    )
        else:
            # If we are not at the first site, we need to take the left index set I_{k-1} and the current index set i_k
            # and compute the total indices left by taking the outer product I_{k-1}⊗i_k. If the index positions are
            # also computed, the np.where function is used to locate the index sets.

            arrL_repeated = np.repeat(self.i[site + 1 - 1], self.grid[site].shape[0], axis=0)
            arrR_tiled = np.tile(self.grid[site], (1, self.i[site + 1 - 1].shape[0])).T

            total_indices_left = np.concatenate((arrL_repeated, arrR_tiled), axis=1)

            if compute_index_pos:
                for current_best_pivot in self.i[site + 1]:
                    current_indices_left_pos.append(
                        np.where(np.all(total_indices_left == current_best_pivot, axis=1))[0][0]
                    )

        # At site k = N-1, there is no righ index set to the right of J_{k + 1}, and we just take all the elements of
        # the last site in the grid.
        if site == self.num_variables - 2:
            total_indices_right = np.array([self.grid[site + 1].copy()], dtype=object).T

            # If we are computing the index positions, we just use the np.where function to find the position of the
            # current best pivot in the total indices set.
            if compute_index_pos:
                for current_best_pivot in self.j[site]:
                    current_indices_right_pos.append(
                        np.where(np.all(total_indices_right == current_best_pivot, axis=1))[0][0]
                    )

        else:
            # If we are not at the last site, we need to take the right index set J_{k+1} and the current index set
            # i_{k+1} and compute the total indices right by taking the outer product J_{k+1}⊗i_{k+1}. If the index
            # positions are also computed, the np.where function is used to locate the index sets.

            arrL_tiled = np.tile(self.grid[site + 1], (1, self.j[site + 1].shape[0])).T
            arrR_repeated = np.repeat(self.j[site + 1], self.grid[site + 1].shape[0], axis=0)

            total_indices_right = np.concatenate((arrL_tiled, arrR_repeated), axis=1)

            if compute_index_pos:
                for current_best_pivot in self.j[site]:
                    current_indices_right_pos.append(
                        np.where(np.all(total_indices_right == current_best_pivot, axis=1))[0][0]
                    )
        if compute_index_pos:
            return total_indices_left, total_indices_right, current_indices_left_pos, current_indices_right_pos
        else:
            return total_indices_left, total_indices_right

    def compute_single_site_tensor(self, site: int) -> np.ndarray:
        """Method that using the index sets stored in the self.i and self.j variables, computes the single-site tensor
        A(I_{k-1}, i_k, J_{k}) used to construct the tensor train.

        Args:
            site (int): The site for which the single-site tensor is computed.

        Returns:
            np.ndarray: The 3-legged tensor corresponding to the tensor at site "site" in the tensor train.
        """
        tensor = np.ndarray((len(self.i[site + 1 - 1]), len(self.grid[site]), len(self.j[site])), dtype=self.f_type)

        # Run over all the points in the set (I_{k-1}, i_k, J_{k})
        for s, left in enumerate(self.i[site + 1 - 1]):
            for k, right in enumerate(self.j[site]):
                for m, i in enumerate(self.grid[site]):

                    # To compute the point given to the function we consider if we are at the first site, the last site
                    # or in the bulk to avoid addind the dummy index at the start or end of the self.i and self.j sets,
                    # respectively.

                    if site == 0:
                        point = np.concatenate(([i], right)).astype(float)
                    elif site == self.num_variables - 1:
                        point = np.concatenate((left, [i])).astype(float)
                    else:
                        point = np.concatenate((left, [i], right)).astype(float)

                    tensor[s, m, k] = self.func(point)

        return tensor

    def compute_cross_blocks(self, site: int) -> np.ndarray:
        """Method that using the index sets stored in the self.i and self.j variables compute the square cross block
        tensors that go in between the 3-leged ones in the tensor train to form the ttcross approximation. The
        tensors are computed as [A(I_{k}, J_{k})]^{-1}.

        Args:
            site (int): The site in the tensor train corresponding to the physical site which lies at the left of this
            inverse cross block.

        Raises:
            ValueError: If the left and right indexes have different dimensions, which results in a non-square block
            which cannot be inverted.

        Returns:
            np.ndarray: The 2-legged tensor that is the inverse of the cross block tensor.
        """
        
        dif = len(self.i[site + 1]) - len(self.j[site])
        dif_in_i = True if dif > 0 else False
        
        if dif != 0:
            warnings.warn(
                f"The left and right indexes must have the same dimension at position {site}. {"I" if dif_in_i else "J"} has {abs(dif)} more elements than {"J" if dif_in_i else "I"} which have been discarded to compute the inverse block."
            )


        # Run over all the points in the set (I_{k}, J_{k}) to compute the block A(I_{k}, J_{k}) and then invert it.
        
        if dif == 0:
            block = np.ndarray((len(self.i[site + 1]), len(self.j[site])), dtype=self.f_type)

            for s, left in enumerate(self.i[site + 1]):
                for k, right in enumerate(self.j[site]):
                    block[s, k] = self.func(np.concatenate((left, right)).astype(float))
          
        else:
            block = np.ndarray(
                (
                    len(self.i[site + 1])-abs(dif) if dif_in_i else len(self.i[site + 1]), 
                    len(self.j[site]) if dif_in_i else len(self.j[site])-abs(dif),
                ), 
                dtype=self.f_type
            )
       
            for s, left in enumerate(self.i[site + 1][:-abs(dif)] if dif_in_i else self.i[site + 1]):
                for k, right in enumerate(self.j[site] if dif_in_i else self.j[site][:-abs(dif)]):
                    block[s, k] = self.func(np.concatenate((left, right)).astype(float))

        inv_block = np.linalg.inv(block)

        return inv_block
    
    def _compute_superblock_tensor_compiled(self, site: int) -> np.ndarray:
        """Helper method that calls the compiled external compute_superblock_tensor function to compute the superblock
        tensor. It is the most expensive part of the algorithm. When the function can indeed be jit compiled, this 
        method is called to compute the superblock tensor.

        Args:
            site (int): _description_

        Returns:
            np.ndarray: _description_
        """
        
        time1 = time.time()
        
        if site == 0:
            s = 0
        elif site == self.num_variables - 1:
            s = 2
        else:
            s = 1

        i = self.i[site].astype(np.float64)
        j = self.j[site + 1].astype(np.float64)
        g1 = self.grid[site].astype(np.float64)
        g2 = self.grid[site + 1].astype(np.float64)

        tensor = compute_superblock_tensor(i, j, g1, g2, s)

        self.func_calls += np.prod(tensor.shape)

        self.super_block_time += time.time() - time1
        
        return tensor

    def _compute_superblock_tensor_non_compiled(self, site: int) -> np.ndarray:
        """Method that using the index sets stored in the self.i and self.j variables computes the superblock tensor
        A(I_{k-1}, i_k, i_{k+1}, J_{k+1}) used to update the index sets in the tensor train in DMRG-like procedures.
        It is the most expensive part of the algorithm. This method is called whenever the function cannot be compiled
        with numba.
        
        Args:
            site (int): The site of the left physical leg of the superblock tensor.

        Returns:
            np.ndarray: The output 4-legged tensor corresponding to the superblock tensor.
        """
        tensor = np.ndarray(
            (len(self.i[site + 1 - 1]), len(self.grid[site]), len(self.grid[site + 1]), len(self.j[site + 1])),
            dtype=self.f_type,
        )

        time1 = time.time()

        # Run over all the points in the set (I_{k-1}, i_k, i_{k+1}, J_{k+1})
        for s, left in enumerate(self.i[site + 1 - 1]):
            for k, right in enumerate(self.j[site + 1]):
                for m, i in enumerate(self.grid[site]):
                    for n, j in enumerate(self.grid[site + 1]):

                        # And as in the single-site tensor, we consider if we are at the first site, the last site or in
                        # the bulk to avoid adding the dummy index at the start or end of the self.i and self.j sets,
                        # respectively.

                        if site == 0:
                            point = np.concatenate(([i], [j], right)).astype(float)
                        elif site == self.num_variables - 1:
                            point = np.concatenate((left, [i], [j])).astype(float)
                        else:
                            point = np.concatenate((left, [i], [j], right)).astype(float)

                        tensor[s, m, n, k] = self.func(point)

        self.super_block_time += time.time() - time1

        return tensor

    @abstractmethod
    def run(self) -> np.ndarray:
        """Run the full algorithm, performing full sweeps until convergence or the maximum number of sweeps is reached.
        After the index sets are updated, the final tensor train built using the computed index set is returned:

        A(i1, i2, ..., iN) ≈
          A(i1, J1) * [A(I1, J1)]^{-1} * A(I1, i2, J2) * [A(I2, J2]^{-1} * ... * [A(I{N-1}, J{N-1)]^{-1} * A(I{N-1}, iN)

        Returns:
            - np.ndarray: The tensor train that contains the ttcross approximation to the tensor related to evaluating
            the function at all the grid points.
        """


class ttrc(tt_interpolator):
    """Class representing the ttrc interpolator presented in:
    D. Savostyanov and I. Oseledets, "Fast adaptive interpolation of multi-dimensional arrays in tensor train format",
    The 2011 International Workshop on Multidimensional (nD) Systems, Poitiers, France, 2011, pp. 1-8,
    doi: 10.1109/nDS.2011.6076873.
    Starting from a large amount of pivots at each site, the algorithm iteratively improves on the selected pivots
    using the maxvol algorithm presented in:
    https://www.researchgate.net/publication/251735015_How_to_Find_a_Good_Submatrix
    Which reduces the number of pivots adaptatively through SVD decompositions.

    In terms of notation, apart from what is clarified in the tt_interpolator parent class,the following extra
    notation is used:
    - self.p[k] = Square matrices obtained after the SVD, QR and maxvol proceses. They are constructed on left to right
    sweeps by selectig the best rows obtained from the maxvol from the left matrix obtained from the orthogonalization
    either by SVD or QR. On right to left sweeps, the best columns are selected from the right matrix outpu of the
    orthogonalization. Their role in the algorithm can be better understood from the paper.

    - self.r = The left matrix obtained from (QR).T on the first warm-up right to left sweep. For more details on its
    rol, refer to the paper.

    - self.b[k] = The 3-legged tensors product of the SVD procedure on the superblock tensor. They are not the tensors
    that form the ttcross approximation of the function, but theintermediate tensors used to check convergence from
    one swipe to the other before computing the new SVD decomposition using the criteria:
    ||C - A|| < tol * ||C|| / sqrt(N-1)

    Args:
        - func (FunctionType): The function to be interpolated. It must be a function that takes a numpy array as input
        and returns a float or complex number. For speed purposes, it is highly recommended to pass a function which can 
        be numba jit compiled.

        - num_variables (int): The number of variables of the grid on which the function is defined.

        - grid (np.ndarray): The grid on which the function is defined. It must be a numpy array of len = num_variables.
            The number of points in each dimension of the grid can be different.

        - maxvol_tol (float): Tolerance for the maxvol algorithm. It is used as 1 + max_tol, as the maxvol algorithm takes
        all the values of an internally defined matrix to values as close as possible to 1.

        - truncation_tol (float): Tolerance for the truncation of the SVD decompositions. The number of singular value
        kept is the number of singular values that add up to a fraction of 1 - truncation_tol of the total sum of
        singular values.

        - sweeps (int): Number of sweeps to perform.

        - initial_bond_guess (int): Initial bond dimension guess. It is the size of all the initial index sets I and J
        stored in the self.i and self.j variables. The selected initial_bond_guess pivots ate each site are selected
        at random, but maintaining the nestedness left and right in all the sites.

        - max_bond (int): Maximum bond dimension allowed in the algorithm. Must be larger than the initial_bond_guess.

        - is_f_complex (bool, optional): Whether the function is complex or not. Defaults to False.
        
        - pivot_initialization (str, optional): The way in which the initial pivots are selected. It can be either
        "random" or "first_n". Defaults to "random".
    """

    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        grid: np.ndarray,
        maxvol_tol: float,
        truncation_tol: float,
        sweeps: int,
        initial_bond_guess: int,
        max_bond: int,
        is_f_complex=False,
        pivot_initialization:str = "random"
    ) -> None:
        super().__init__(func, num_variables, grid, maxvol_tol, sweeps, is_f_complex, pivot_initialization)

        if initial_bond_guess > max([grid_dim.shape[0] for grid_dim in grid]):
            self.init_bond = max([grid_dim.shape[0] for grid_dim in grid])
            warnings.warn(
                "The initial bond guess is larger than the maximum grid size. Max bond gets initialized to the maximum of grid size instead of given initial_bond_guess."
            )
        else:
            self.init_bond = initial_bond_guess

        if max_bond < initial_bond_guess:
            raise ValueError("Maximum allowed bond dimension must be larger than the initial bond guess.")
        else:
            self.max_bond = max_bond

        self._create_initial_index_sets()

        # When picking the initial index sets at random, we can fall into numpy erros easily (this also
        # happens in the deterministic case). With the folloing while loop, we make sure that the initial index sets
        # do not raise error and if they do, we repeat the initialization process.
        
        check_initialization_singularity = True

        time1 = time.time()
        tries = 1

        if self.pivot_init == "random":
            while check_initialization_singularity:
                try:
                    self._create_initial_arrays()
                    check_initialization_singularity = False

                except np.linalg.LinAlgError:
                    self._create_initial_index_sets()
                    tries += 1
        else:
            self._create_initial_arrays()

        self._create_initial_bond_dimensions()
        self.trunctol = truncation_tol

        time2 = time.time()
        print(f"Initialization done after time: {time2 - time1} seconds and {tries} tries.")

    def _create_initial_index_sets_random(self) -> None:
        """Method that creates the initial index sets for all the sites in the tensor train by taking pivots at random
        while maintaning the left and right nestedness of I and J, respectively. The number of pivots taken at each site
        follows the criteria:
         1. At the current site, take min(len(grid[site]), max_bond) pivots.
         2. Compare if this number of pivots is smaller/bigger than the number of pivots at the left/right site,
            depending on if we are working with the I or J indices, respectively.
         3. If it is smaller, expand the currently selected points by repeating them until they reach the same
            dimension as the left/right index set.
         4. If it is bigger, repeat the left/right index set until it reaches the same dimension as the currently
            selected points.

        In this way, the nestedness of the index sets is maintained, the pivots are not repeated (which is fatal for
        the maxvol algorithm, as it would make the matrix singular) and the number of pivots is adapted to each
        site without ever exceeding the maximum bond dimension.
        """
        np.random.seed(0)
        self.i = np.ndarray(self.num_variables, dtype=object)

        # Add first the dummy index set in the first position of self.i and create the first real index set at site 1
        # by taking min(len(grid[0]), max_bond) pivots at random.
        self.i[0] = np.array([[1]])
        self.i[1] = np.array(
            [np.random.choice(self.grid[0], size=min(len(self.grid[0]), self.init_bond), replace=False)]
        ).T
        for k in range(2, self.num_variables):
            # At site k-1 (we are indexing according to self.i, which is 1 site ahead the site indexing in the grid),
            # take min(len(grid[k-1]), max_bond) pivots at random.
            current_index = np.array(
                [np.random.choice(self.grid[k - 1], size=min(len(self.grid[k - 1]), self.init_bond), replace=False)]
            ).T

            # If the number of pivots taken at the current site is smaller than the number of pivots taken at the
            # previous site, take a random sample of the previous index set to match the current one and stack them.
            if len(current_index) <= len(self.i[k - 1]):
                previous_selected_rows = np.random.choice(len(self.i[k - 1]), size=len(current_index), replace=False)
                previous_indices = self.i[k - 1][previous_selected_rows]
                self.i[k] = np.column_stack((previous_indices, current_index))

            # If the number of pivots taken at the current site is bigger than the number of pivots taken at the
            # previous site, repeat the previous index set until it reaches the same dimension as the current one, and
            # then stack them.
            else:
                times = len(current_index[0]) // len(self.i[k - 1][0])
                previous_choice = np.row_stack([self.i[k - 1] for _ in range(times + 1)])
                self.i[k] = np.column_stack((previous_choice[: len(current_index)], current_index))

        # =======================================================================================================
        # Repeat the same process for the right index sets J starting from the last site and running left.

        self.j = np.ndarray(self.num_variables, dtype=np.ndarray)
        self.j[-1] = np.array([[1]])
        self.j[-2] = np.array(
            [np.random.choice(self.grid[-1], size=min(len(self.grid[-1]), self.init_bond), replace=False)]
        ).T

        for k in range(-3, -self.num_variables - 1, -1):
            current_index = np.array(
                [np.random.choice(self.grid[k + 1], size=min(len(self.grid[k + 1]), self.init_bond), replace=False)]
            ).T

            if len(current_index) <= len(self.j[k + 1]):
                selected_columns = np.random.choice(len(self.j[k + 1]), size=len(current_index), replace=False)
                previous_choice = self.j[k + 1][selected_columns]
                self.j[k] = np.column_stack((current_index, previous_choice))
            else:
                times = len(current_index) // len(self.j[k + 1])
                previous_choice = np.column_stack(([self.j[k + 1] for _ in range(times + 1)]))
                self.j[k] = np.column_stack((current_index, previous_choice[: len(current_index)]))

    # Optional method to creat the initial index sets by taking the first element that appear in each dimension of the
    # grid, instead of picking at random.

    def _create_initial_index_sets_firstn(self):
        """Method that creates the initial index sets for all the sites in the tensor train by taking pivots at random
        while maintaning the left and right nestedness of I and J, respectively. The number of pivots taken at each site
        follows the criteria:
         1. At the current site, take min(len(grid[site]), init_bond) pivots.
         2. Compare if this number of pivots is smaller/bigger than the number of pivots at the left/right site,
            depending on if we are working with the I or J indices, respectively.
         3. If it is smaller, expand the currently selected points by repeating them until they reach the same
            dimension as the left/right index set.
         4. If it is bigger, repeat the left/right index set until it reaches the same dimension as the currently
            selected points.

        In this way, the nestedness of the index sets is maintained, the pivots are not repeated (which is fatal for
        the maxvol algorithm, as it would make the matrix singular) and the number of pivots is adapted to each
        site without ever exceeding the maximum bond dimension.
        """
        self.i = np.ndarray(self.num_variables, dtype=object)

        # Add first the dummy index set in the first position of self.i and create the first real index set at site 1
        # by taking min(len(grid[0]), init_bond) pivots at random.
        self.i[0] = np.array([[1]])
        self.i[1] = np.array([self.grid[0][: min(len(self.grid[0]), self.init_bond)]]).T

        for k in range(2, self.num_variables):
            # At site k-1 (we are indexing according to self.i, which is 1 site ahead the site indexing in the grid),
            # take min(len(grid[k-1]), init_bond) pivots at random.
            current_index = np.array([self.grid[k - 1][: min(len(self.grid[k - 1]), self.init_bond)]]).T

            # If the number of pivots taken at the current site is smaller than the number of pivots taken at the
            # previous site, take a random sample of the previous index set to match the current one and stack them.
            if len(current_index) <= len(self.i[k - 1]):
                previous_indices = self.i[k - 1][: len(current_index)]
                self.i[k] = np.column_stack((previous_indices, current_index))

            # If the number of pivots taken at the current site is bigger than the number of pivots taken at the
            # previous site, repeat the previous index set until it reaches the same dimension as the current one, and
            # then stack them.
            else:
                times = len(current_index[0]) // len(self.i[k - 1][0])
                previous_choice = np.row_stack([self.i[k - 1] for _ in range(times + 1)])
                self.i[k] = np.column_stack((previous_choice[: len(current_index)], current_index))

        # =======================================================================================================
        # Repeat the same process for the right index sets J starting from the last site and running left.

        self.j = np.ndarray(self.num_variables, dtype=np.ndarray)
        self.j[-1] = np.array([[1]])
        self.j[-2] = np.array([self.grid[-1][: min(len(self.grid[-1]), self.init_bond)]]).T

        for k in range(-3, -self.num_variables - 1, -1):
            current_index = np.array([self.grid[k + 1][: min(len(self.grid[k + 1]), self.init_bond)]]).T

            if len(current_index) <= len(self.j[k + 1]):
                previous_choice = self.j[k + 1][: len(current_index)]
                self.j[k] = np.column_stack((current_index, previous_choice))
            else:
                times = len(current_index) // len(self.j[k + 1])
                previous_choice = np.column_stack(([self.j[k + 1] for _ in range(times + 1)]))
                self.j[k] = np.column_stack((current_index, previous_choice[: len(current_index)]))
                
    def _create_initial_index_sets(self) -> None:
        """Helper method to call the pivot initialization methods depending on if the user wants to initialize the
        pivots at random or by taking the first n points in each dimension of the grid.
        """
        if self.pivot_init == "random":
            self._create_initial_index_sets_random()
        elif self.pivot_init == "first_n":
            self._create_initial_index_sets_firstn()
        else:
            raise ValueError("Pivot initialization must be either 'random' or 'first_n'.")
            

    def _create_initial_bond_dimensions(self) -> None:
        """Method that initializes the bond dimensions for all the sites in the tensor train. The bond dimensions are
        computed from the sizes of the index sets I and J at each site. The bond dimensions are stored in the
        self.bonds variable.

        Raises:
            - ValueError: If the initial left and right index sets have different dimensions, which would result in non
            square blocks that cannot be inverted.
        """
        self.bonds = np.ndarray(self.num_variables - 1, dtype=int)
        for k in range(self.num_variables - 1):
            if self.i[k + 1].shape[0] != self.j[k].shape[0]:
                raise ValueError("Initial left and right indexes should have the same dimension")
            self.bonds[k - 1] = self.i[k + 1].shape[0]

    def _create_initial_arrays(self) -> None:
        """Method that creates all the initial tensors needed by the algorithm as described in the original ttrc paper.
        This includes:
        - The initial P matrix array, as well as initializing the first and las ones to identity matrices os shape (1,1).
        - The initial R matrix.
        - The initial index sets I and J.
        - The initial bond dimensions.
        - The initial crude approximation to A, built from the B tensors computed from the randomly chosen pivots.
        """
        self.p = np.ndarray(self.num_variables + 1, dtype=np.ndarray)
        self.p[0] = np.array([[1]], dtype=np.float_)
        self.p[-1] = np.array([[1]], dtype=np.float_)
        self.r = np.array([[1]], dtype=np.float_)

        self.b = np.ndarray(self.num_variables, dtype=np.ndarray)
        self.b[0] = self.compute_single_site_tensor(0)

        for i in range(1, self.num_variables):
            self.b[i] = self.compute_single_site_tensor(i)

    def initial_sweep(self) -> None:
        """Method that performs the initial warm-up sweep from right to left following step by step the procedure
        described in the original ttrc paper. The main steps are:
        1. Contract to the current tensor B at site k the matrix R from thr right.
        2. Perform a QR decomposition on the resulting tensor and transpose the whole thing such that the matrix with
        orthogonal columns is on the right and the upper diagonal on the left, the latter updating the R matrix.
        3. Contract into the Q matrix the P matrix from the right. and apply the Maxvol to it.
        4. Update the random index set J at site k-1 with the selected pivots by the Maxvol, and update the P matrix
            at site k-1 with the selected columns from the Q matrix (There's a typo on the paper, where they say
            P_{k-1} = Q[:, I_k], but it should be P_{k-1} = Q[:, J_k]).
        """
        for pos in range(self.num_variables - 1, 0, -1):
            # Obtain the shape of the current B tensor at site k.
            rk_1 = self.b[pos].shape[0]
            nk = self.b[pos].shape[1]
            rk = self.b[pos].shape[2]

            # Contract the current B tensor at site k with the R matrix from the right.
            self.b[pos] = ncon([self.b[pos], self.r], [[-1, -2, 1], [1, -3]])

            # Reshape the resulting tensor to perform the QR decomposition.
            c = np.reshape(self.b[pos], (rk_1, nk * rk))
            q_T, r_T = np.linalg.qr(c.T)

            q = q_T.T
            self.r = r_T.T

            # Contract the P matrix from the right into the Q matrix.
            self.b[pos] = np.reshape(q, (rk_1, nk, rk))

            q = np.reshape(q, (rk_1 * nk, rk))

            q = q @ self.p[pos + 1]

            q = np.reshape(q, (rk_1, nk * rk))

            # Obtain the total indices set I_{k-1}⊗i_k and J_{k+1}⊗j_{k+1} for the current site k and perfrom the
            # maxvol algorithm to update the index set J at site k-1.
            _, J_k_1_expanded = self._obtain_superblock_total_indices(site=pos - 1, compute_index_pos=False)

            best_indices, self.j[pos - 1] = py_maxvol(
                A=q, full_index_set=J_k_1_expanded, tol=1 + self.tol, max_iters=10000
            )

            # Update the P maatrix to the left of the current site k with the selected columns from the Q matrix.
            self.p[pos] = q[:, best_indices]

        # For the last one, we just contract the R matrix from the right and update the B tensor.
        self.b[0] = ncon([self.b[0], self.r], [[-1, -2, 1], [1, -3]])

    def check_convergence(self, C: np.ndarray, site: int) -> None:
        """Checks if the algorithm has converged by comparing the current superblock tensor C with the approximation
        obtained in the previous sweep. The convergence criteria is:
        ||C - B_{site} * B_{site+1}|| < tol * ||C|| / sqrt(N-1)

        Args:
            - C (np.ndarray): The current superblock tensor A(I_{k-1}, i_k, i_{k+1}, J_{k+1}).
            - site (int): The site of the left physical leg of the superblock.
        """
        approx_superblock = ncon([self.b[site], self.b[site + 1]], [[-1, -2, 1], [1, -3, -4]])
        err = np.linalg.norm(
            np.reshape(C - approx_superblock, (C.shape[0] * C.shape[1], C.shape[2] * C.shape[3])), ord="fro"
        )
        bound = (
            self.trunctol
            * np.linalg.norm(np.reshape(C, (C.shape[0] * C.shape[1], C.shape[2] * C.shape[3])), ord="fro")
            / np.sqrt(self.num_variables - 1)
        )
        self.not_converged = err > bound
        # print(err)

    def left_right_update(self, site) -> None:
        """Method that performs a singe update on a left to right sweep. The main steps follow the exact same structure
        as what is described in the original ttrc paper, and are the following:
        1. Compute the superblck tensor C = A(I_{k-1}, i_k, i_{k+1}, J_{k+1}) at site k.
        2. Contract the inverses of the P matrices from the left and right into the superblock tensor C. This is to
        be able to compare with the B matrices from the previous sweep.
        3. Check the convergence using the check_convergence method.
        4. Perform an SVD decomposition on the reshaped superblock tensor C and update the B matrices at site k and k+1.
        5. Recontract the P matrix from the left (to update the bond dimension it is not needed) and perfrom the maxvol
        algorithm to update the I index set at site k.
        6. Update the P matrix at site k+1 with the selected ows from the left matrix of the SVD.

        Args:
            - site (int): The site corresponding to the left physical leg of the superblock tensor.
        """

        # Compute the superblock tensor C = A(I_{k-1}, i_k, i_{k+1}, J_{k+1}) at site k and store the bond dimensions.
            
        C = self.compute_superblock_tensor(site)
        r_left = C.shape[0]
        r_right = C.shape[3]

        # Compute the inverses of the P matrices at left and right and contract them into the superblock tensor C.
        inv_prev_p = np.linalg.inv(self.p[site])
        inv_post_p = np.linalg.inv(self.p[site + 2])

        C = ncon(
            [inv_prev_p, C, inv_post_p],
            [[-1, 1], [1, -2, -3, 2], [2, -4]],
        )

        # Check the convergence of the algorithm.
        self.check_convergence(C, site)

        # Reshape the C matrix and perform the SVD decomposition, truncating the number of singular values kept with
        # the tolerance trunctol.
        C_reshaped = np.reshape(C, (r_left * len(self.grid[site]), len(self.grid[site + 1]) * r_right))

        # utemp, s_temp, vtemp = np.linalg.svd(C_reshaped, full_matrices=True)
        utemp, s_temp, vtemp = svd(C_reshaped, full_matrices=True)

        stemp_cumsum = np.cumsum(s_temp)
        chitemp = min(np.argmax(stemp_cumsum >= (1.0 - self.trunctol) * stemp_cumsum[-1]) + 1, self.max_bond)

        utemp = utemp[:, :chitemp]
        vtemp = vtemp[:chitemp, :]
        stemp = np.diag(s_temp[:chitemp])

        # Update the B matrices at site k and k+1 with the output of the SVD decomposition and prepare the left tensor
        # for the maxvol by contracting the P matrix from the left again into the left block of the SVD.
        self.b[site] = np.reshape(utemp, (r_left, len(self.grid[site]), chitemp))
        right_tensor = ncon([stemp, vtemp], [[-1, 1], [1, -2]])
        self.b[site + 1] = np.reshape(right_tensor, (chitemp, len(self.grid[site + 1]), r_right))

        self.bonds[site] = chitemp

        maxvol_matrix = ncon([self.p[site], self.b[site]], [[-1, 1], [1, -2, -3]])
        maxvol_matrix = np.reshape(maxvol_matrix, (r_left * len(self.grid[site]), chitemp))

        # Obtain the total indices set I_{k-1}⊗i_k for the current site k and perfrom the maxvol
        # to update the index set I at site "site".
        I_k_1_expanded, _ = self._obtain_superblock_total_indices(site, compute_index_pos=False)

        best_indices, self.i[site + 1] = py_maxvol(
            A=maxvol_matrix, full_index_set=I_k_1_expanded, tol=1 + self.tol, max_iters=100000
        )

        # Update the P matrix to the right of the current site and to the left of site+1 with the selected rows from the
        # left matrix of the SVD.
        self.p[site + 1] = maxvol_matrix[best_indices]

    def right_left_update(self, site) -> None:
        """Method that performs a singe update on a right to left sweep. The main steps follow the exact same structure
        as what is described in the original ttrc paper for the left to right sweep, but in reverse, and are the
        following:
        1. Compute the superblck tensor C = A(I_{k-1}, i_k, i_{k+1}, J_{k+1}) at site k.
        2. Contract the inverses of the P matrices from the left and right into the superblock tensor C. This is to
        be able to compare with the B matrices from the previous sweep.
        3. Check the convergence using the check_convergence method.
        4. Perform an SVD decomposition on the reshaped superblock tensor C and update the B matrices at site k and k+1.
        5. Recontract the P matrix from the right (to update the bond dimension it is not needed) and perfrom the maxvol
        algorithm to update the J index set at site k + 1.
        6. Update the P matrix at site k+1 with the selected columns from the right matrix of the SVD.

        Args:
            - site (int): The site corresponding to the left physical leg of the superblock tensor.
        """

        # Compute the superblock tensor C = A(I_{k-1}, i_k, i_{k+1}, J_{k+1}) at site k and store the bond dimensions.
        C = self.compute_superblock_tensor(site)
        r_left = C.shape[0]
        r_right = C.shape[3]

        # Compute the inverses of the P matrices at left and right and contract them into the superblock tensor C.
        inv_prev_p = np.linalg.inv(self.p[site])
        inv_post_p = np.linalg.inv(self.p[site + 2])

        C = ncon(
            [inv_prev_p, C, inv_post_p],
            [[-1, 1], [1, -2, -3, 2], [2, -4]],
        )

        # Check the convergence of the algorithm.
        self.check_convergence(C, site)

        # Reshape the C matrix and perform the SVD decomposition, truncating the number of singular values kept with
        # the tolerance trunctol.
        C_reshaped = np.reshape(C, (r_left * len(self.grid[site]), len(self.grid[site + 1]) * r_right))

        # utemp, s_temp, vtemp = np.linalg.svd(C_reshaped, full_matrices=True)
        utemp, s_temp, vtemp = svd(C_reshaped, full_matrices=True)

        stemp_cumsum = np.cumsum(s_temp)
        chitemp = min(np.argmax(stemp_cumsum >= (1 - self.trunctol) * stemp_cumsum[-1]) + 1, self.max_bond)

        utemp = utemp[:, :chitemp]
        vtemp = vtemp[:chitemp, :]
        stemp = np.diag(s_temp[:chitemp])

        # Update the B matrices at site k and k+1 with the output of the SVD decomposition and prepare the right tensor
        # for the maxvol by contracting the P matrix from the right again into the right block of the SVD.
        left_tensor = ncon([utemp, stemp], [[-1, 1], [1, -2]])
        self.b[site] = np.reshape(left_tensor, (r_left, len(self.grid[site]), chitemp))
        self.b[site + 1] = np.reshape(vtemp, (chitemp, len(self.grid[site + 1]), r_right))

        self.bonds[site] = chitemp

        maxvol_matrix = ncon([self.b[site + 1], self.p[site + 2]], [[-1, -2, 1], [1, -3]])
        maxvol_matrix = np.reshape(maxvol_matrix, (chitemp, len(self.grid[site + 1]) * r_right))

        # Obtain the total indices set J_{k+1}⊗i_{k+1} for the current site k and perfrom the maxvol
        # to update the index set I at site "site".
        _, J_k_1_expanded = self._obtain_superblock_total_indices(site, compute_index_pos=False)
        best_indices, self.j[site] = py_maxvol(A=vtemp, full_index_set=J_k_1_expanded, tol=1 + self.tol, max_iters=100000)

        # Update the P matrix to the right of the current site and to the left of site+1 with the selected rows from the
        # left matrix of the SVD.
        self.p[site + 1] = vtemp[:, best_indices]

    def full_sweep(self) -> None:
        """Perform a full left to right and right to left sweep in the tensor train."""
        for site in range(self.num_variables - 1):
            self.left_right_update(site)

        for site in range(self.num_variables - 2, -1, -1):
            self.right_left_update(site)
            
    def check_index_sets(self):
        """At the end of the algorithm, checks if all the index sets I_k and J_k have the same size in order to build 
        inverse blocks that form the approximation. If they don't, the method discards the last pivots in the index
        sets, either in I or J, in order to make them the same size.
        """
        self.non_truncated_i = self.i.copy()
        self.non_truncated_j = self.j.copy()

        for site in range(self.num_variables - 1):
            dif = len(self.i[site + 1]) - len(self.j[site])
            dif_in_i = True if dif > 0 else False
            
            if dif != 0:
                self.i[site + 1] = self.i[site + 1][: -abs(dif)] if dif_in_i else self.i[site + 1]
                self.j[site] = self.j[site][: -abs(dif)] if not dif_in_i else self.j[site]
                
    def run(self) -> np.ndarray:
        """Run the full algorithm, performing the initial sweep and then the full sweeps until convergence or the
        maximum number of sweeps is reached. After the index sets are updated, the final tensor train built using the
        computed index set is returned:

        A(i1, i2, ..., iN) ≈
          A(i1, J1) * [A(I1, J1)]^{-1} * A(I1, i2, J2) * [A(I2, J2]^{-1} * ... * [A(I{N-1}, J{N-1)]^{-1} * A(I{N-1}, iN)

        Returns:
            - np.ndarray: The tensor train that contains the ttcross approximation to the tensor related to evaluating
            the function at all the grid points.
        """
        self.total_time = time.time()
        self.initial_sweep()
        for sweep in range(self.sweeps):
            print("Sweep", sweep + 1)
            self.not_converged = False
            self.full_sweep()

            if not self.not_converged:
                print(f"Converged at sweep {sweep}")
                break

        mps = np.ndarray(2 * self.num_variables - 1, dtype=np.ndarray)

        self.check_index_sets()
        
        for site in range(self.num_variables - 1):
            mps[2 * site] = self.compute_single_site_tensor(site)
            mps[2 * site + 1] = self.compute_cross_blocks(site)

        mps[-1] = self.compute_single_site_tensor(self.num_variables - 1)
        self.total_time = time.time() - self.total_time
        return mps


class greedy_cross(tt_interpolator):
    """Class representing the greedy ttcross algorithm presented in:
    Dmitry V. Savostyanov, "Quasioptimality of maximum-volume cross interpolation of tensors",
    Linear Algebra and its Applications, Volume 458, 2014, Pages 217-244, ISSN 0024-3795,
    https://doi.org/10.1016/j.laa.2014.06.006.
    The algorithm starts with a single pivot at each site and iteratively adds new pivots in a greedy manner.

    Args:
        - func (FunctionType): The function to be interpolated. It must be a function that takes a numpy array as input
        and returns a float or complex number.

        - num_variables (int): The number of variables of the grid on which the function is defined.

        - grid (np.ndarray): The grid on which the function is defined. It must be a numpy array of len = num_variables.
            The number of points in each dimension of the grid can be different.

        - tol (float): The tolerance used in the pivot update to check if the difference between the full block and
        the approximation does not contain any value larger than tol.

        - max_bond (int): The maximum bond dimension allowed in the algorithm.

        - sweeps (int): Number of sweeps to perform.

        - is_f_complex (bool, optional): Whether the function is complex or not. Defaults to False.
        
        - pivot_initialization (str, optional): The method used to initialize the pivots at each site. It can be either
            'random' or 'first_n'. Defaults to 'random'.
    """

    def __init__(
        self,
        func: FunctionType,
        num_variables: int,
        grid: np.ndarray,
        tol: float,
        max_bond: int,
        sweeps: int,
        is_f_complex=False,
        pivot_initialization:str="random",
    ) -> None:
        super().__init__(func, num_variables, grid, tol, sweeps, is_f_complex, pivot_initialization)
        self.max_bond = max_bond
        self._create_initial_index_sets()
        self._create_initial_bonds()
        self.error = []

        self._create_initial_index_sets()

        # When picking the initial index sets at random, we can fall into Singular matrices very easily (this also
        # happens in the deterministic case). With the folloing while loop, we make sure that the initial index sets
        # are not singular, and if they are, we repeat the process until we find a non-singular set of index sets.

        # IMPORTANT: IF THE INITIALIZATION IS SWITCHED TO DETERMINISTIC, THIS WHILE LOOP ILL NEVER END.
        check_initialization_singularity = True

        time1 = time.time()
        tries = 1

        if self.pivot_init == "random":
            while check_initialization_singularity and tries < 1000:
                try:
                    for site in range(self.num_variables - 1):
                        _ = self.compute_cross_blocks(site)
                    check_initialization_singularity = False

                except np.linalg.LinAlgError:
                    self._create_initial_index_sets()
                    tries += 1
                    
                if tries == 1000:
                    raise ValueError("Initialization failed after 1000 tries. Try again.")
        else:
            try:
                for site in range(self.num_variables - 1):
                    _ = self.compute_cross_blocks(site)
            
            except np.linalg.LinAlgError:
                raise ValueError(
                    "Initialization with the first n points results in singular matrices. Try with the random initialization."
                )

        time2 = time.time()

        print(f"Initialization succesfully done after time: {time2 - time1} seconds and {tries} tries.")

    def _create_initial_index_sets_random(self) -> None:
        """Method that creates the initial index sets for all the sites in the tensor train by taking a single pivot at
        each site. The pivot is taken by concatenating a random element of the grid at the current site to the
        index set at the left/right site for the self.i/self.j variables, respectively.
        """
        np.random.seed(10)
        self.i = np.ndarray(self.num_variables + 1, dtype=object)

        # Add first the dummy index set in the first position of self.i and create the first real index set at site 1
        # by taking a random pivot at the first site. For the next ones, simply append to the pivot to the left
        # a randomly selected point from the grid at the current site, maintaining the nestedness of the index sets.
        self.i[0] = np.array([[1.0]])
        self.i[1] = np.array([[np.random.choice(self.grid[0])]])
        for i in range(2, self.num_variables + 1):
            current_index = np.array([np.random.choice(self.grid[i - 1])])
            self.i[i] = np.column_stack((self.i[i - 1], current_index))

        # Repeat the same thing for the right index sets J starting from the last site and running left.
        self.j = np.ndarray(self.num_variables, dtype=object)
        self.j[-1] = np.array([[1.0]])
        self.j[-2] = np.array([[np.random.choice(self.grid[-1])]])
        for i in range(-3, -self.num_variables - 1, -1):
            current_index = np.array([np.random.choice(self.grid[i + 1])])
            self.j[i] = np.column_stack((current_index, self.j[i + 1]))

    # Optional method to create the initial index sets by taking the first point of the grid at each site, instead of
    # taking random pivots

    def _create_initial_index_sets_firstn(self) -> None:
        """Method that creates the initial index sets for all the sites in the tensor train by taking a single pivot at
        each site. The pivot is taken by concatenating the first element of the grid at the current site to the
        index set at the left/right site for the self.i/self.j variables, respectively.
        """
        self.i = np.ndarray(self.num_variables + 1, dtype=object)

        # Add first the dummy index set in the first position of self.i and create the first real index set at site 1
        # by taking the first point in the grid at the first site. For the next ones, simply append to the pivot to the
        # left a the first point from the grid at the current site, maintaining the nestedness of the index sets.
        self.i[0] = np.array([[1.0]])
        self.i[1] = np.array([[self.grid[0][0]]])
        for i in range(2, self.num_variables + 1):
            current_index = np.array([self.grid[i - 1][0]])
            self.i[i] = np.column_stack((self.i[i - 1], current_index))

        # Repeat the same thing for the right index sets J starting from the last site and running left.
        self.j = np.ndarray(self.num_variables, dtype=object)
        self.j[-1] = np.array([[1.0]])
        self.j[-2] = np.array([[self.grid[-1][0]]])
        for i in range(-3, -self.num_variables - 1, -1):
            current_index = np.array([self.grid[i + 1][0]])
            self.j[i] = np.column_stack((current_index, self.j[i + 1]))
            
    def _create_initial_index_sets(self) -> None:
        """Helper method to call the pivot initialization methods depending on if the user wants to initialize the
        pivots at random or by taking the first n points in each dimension of the grid.
        """
        if self.pivot_init == "random":
            self._create_initial_index_sets_random()
        elif self.pivot_init == "first_n":
            self._create_initial_index_sets_firstn()
        else:
            raise ValueError("Pivot initialization must be either 'random' or 'first_n'.")

    def _create_initial_bonds(self) -> None:
        """Method that initializes the bond dimensions of the tensor train to 1 at all sites."""
        self.bonds = np.ones(self.num_variables - 1, dtype=int)

    def index_update(self, site: int) -> None:
        """Method that performs an index update on a superblock at sites "site" and "site + 1". The main steps are:
        1. Check if the maximum bond dimension has been reached at the current site. If it has, return without updating
        the index set.
        2. Compute the superblock tensor A(I_{k-1}, i_k, i_{k+1}, J_{k+1}) at site k and reshape it to a matrix.
        3. Obtain the sets I_1i = I_{k-1}⊗i_k and J_1j = J_{k+1}⊗i_{k+1} and the postion on these sets of the
        current index set I and J that define the best approximation so far.
        4. Compute the new index sets I and J by using the greedy_pivot_finder algorithm, which expands the current
        index set by adding the pivot that reduces the error the most.
        5. Update the index sets I and J, the bond dimension at site k and the error list with the new error between the
        matrix obtained from the full index sets I_{k-1}⊗i_k and J_{k+1}⊗i_{k+1} and the approximation
        with the index sets I_k and J_k.

        Args:
            site (int): The left site of the superblock tensor.
        """

        # Check if the maximum bond dimension has been reached at the current site. If it has, return without updating.
        if self.bonds[site] == self.max_bond:
            return

        # Compute the superblock tensor at site k and reshape it to a matrix.
            
        superblock_tensor = self.compute_superblock_tensor(site)
        
        superblock_tensor = np.reshape(
            superblock_tensor,
            (
                superblock_tensor.shape[0] * superblock_tensor.shape[1],
                superblock_tensor.shape[2] * superblock_tensor.shape[3],
            ),
        )

        # Obtain the total indices set I_{k-1}⊗i_k and J_{k+1}⊗i_{k+1} for the current site k and the position of the
        # current index set I and J in these sets and update them with a new pivot using the greedy_pivot_finder.
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

    def full_sweep(self) -> None:
        """Perform a full left to right and right to left sweep in the tensor train."""
        for site in range(self.num_variables - 1):
            self.index_update(site)

        for site in range(self.num_variables - 2, -1, -1):
            self.index_update(site)

    def run(self) -> np.ndarray:
        """Execute the full greedy ttcross algorithm. The algorithm performs the sweeps until the maximum number of
        sweeps is reached or when the bond dimensions are not updated after a full sweep, which means that either the
        maximum bond dimension has been reached or the algorithm has converged. After the final index sets are computed,
        the final tensor train is built using the computed index sets as:
        A(i1, i2, ..., iN) ≈
          A(i1, J1) * [A(I1, J1)]^{-1} * A(I1, i2, J2) * [A(I2, J2]^{-1} * ... * [A(I{N-1}, J{N-1)]^{-1} * A(I{N-1}, iN)

        Returns:
            np.ndarray: The tensor train that contains the ttcross approximation to the tensor related to evaluating
            the function at all the grid points.
        """

        self.total_time = time.time()
        for s in range(self.sweeps):
            # Save the current bond dimensions to check if they have been updated after the sweep. If not, the algorithm
            # has converged and we can stop.
            print("Sweep", s + 1)
            pre_sweep_bonds = self.bonds.copy()
            self.full_sweep()
            if np.array_equal(pre_sweep_bonds, self.bonds):
                break

        mps = np.ndarray(2 * self.num_variables - 1, dtype=np.ndarray)

        for site in range(self.num_variables - 1):
            mps[2 * site] = self.compute_single_site_tensor(site)
            mps[2 * site + 1] = self.compute_cross_blocks(site)

        mps[-1] = self.compute_single_site_tensor(self.num_variables - 1)

        self.total_time = time.time() - self.total_time
        return mps

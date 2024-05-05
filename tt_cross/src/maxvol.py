import numpy as np
import numba as nb
from scipy.linalg import get_blas_funcs, get_lapack_funcs, lu_factor


# @nb.jit(nopython=True)
def maxvol(A: np.ndarray, tol: float, max_iter: int) -> tuple[list[int], np.ndarray]:
    """Maxvol algorithm, which finds the subset of rows of a matrix A which form a submatrix with the largest volume.
    Given a matrix A of shape (n, r) with n > r, the algorithm returns a list of r indices of rows of A. It uses a
    sequential procedure that swaps rows to maximize the volume. The reference paper can be found in:
    https://www.researchgate.net/publication/251735015_How_to_Find_a_Good_Sub

    The algorithm follows the steps below:
    - The input matrix will be renamed to B, which will be a copy of A, transposed if n < r.
    - The maximal submatrix will be C, which initially will be asumed to lie in the first r rows of B.
                                  | I |
    - We know that B * C^-1 = D = |   | , where I is an identity of shape (r, r).
                                  | Z |
    - We will solve the system C^T * X = B^T, where X = D^T. This will give us the matrix D.
    - And with D, we can just find max(D[i, j]), add the swap i<->j to the list of indices, and update D with the
    procedure described in the paper, with an update to just the Z part of D.

    Args:
        A (np.ndarray): The input matrix of shape (n, r). If n < r, the algorithm will transpose the matrix.
        tol (float): The tolerance 1+delta that the algorithm will use to check for convergence.
        max_iter (int): A maximum number of iterations that the algorithm will perform.

    Raises:
        ValueError: If the input matrix A is not a 2D matrix.

    Returns:
        list[int]: A list of r indices of rows of A that form a submatrix with the largest volume.
    """

    # Initial checks about the input matrix
    if len(A.shape) != 2:
        raise ValueError("A must be a matrix")

    if A.shape[0] < A.shape[1]:
        B = A.T.copy()
    else:
        B = A.copy()

    n, r = B.shape

    # Maxvol works best if the initial pivots are somewhat good. On the paper they suggest using the
    # row pivots from Gaussian elimination, which are very closely related to the LU decomposition ones.
    H, piv = lu_factor(B)
    index = np.arange(n, dtype=np.int32)

    for i in range(r):
        tmp = index[i]
        index[i] = index[piv[i]]
        index[piv[i]] = tmp

    print(index)

    C = H[:r]

    # Solve the system C^T * X = H^T
    D = np.linalg.solve(C.T, H.T).T

    # trtrs = get_lapack_funcs("trtrs", [C])
    # trtrs(C, B.T, trans=1, lower=0, unitdiag=0, overwrite_b=1)
    # trtrs(C, B.T, trans=1, lower=1, unitdiag=1, overwrite_b=1)

    iter = 1

    # FInd the first swap

    i, j = divmod(np.argmax(np.abs(D)), r)

    # print(D)

    while np.abs(D[i, j]) > tol and iter < max_iter:
        # Perform the swap on the index list
        # tmp = index[i]
        # index[i] = index[j]
        # index[j] = tmp

        index[i] = j

        # Update the bottom part of the matrix D accordingly and repeat
        tmp_row = D[i].copy()
        tmp_column = D[:, j].copy()
        tmp_column[i] -= 1.0
        tmp_row[j] += 1.0
        coeff = -1.0 / D[i, j]

        D[r:] = D[r:] + coeff * np.outer(tmp_column, tmp_row)[r:]
        i, j = divmod(np.argmax(np.abs(D)), r)

        iter += 1

    return index[:r].copy(), D


def py_maxvol(A: np.ndarray, full_index_set: np.ndarray, tol=1.05, max_iters=100):
    """
    Python implementation of 1-volume maximization.

    See Also
    --------
    maxvol
    """
    # some work on parameters
    if tol < 1:
        tol = 1.0
    if A.shape[0] < A.shape[1]:
        A = A.T
    N, r = A.shape

    B = np.copy(A[:N], order="F")
    C = np.copy(A.T, order="F")
    H, ipiv, info = get_lapack_funcs("getrf", [B])(B, overwrite_a=1)

    index = np.arange(N, dtype=np.int32)
    for i in range(r):
        tmp = index[i]
        index[i] = index[ipiv[i]]
        index[ipiv[i]] = tmp

    # solve A = CH, H is in LU format
    B = H[:r]

    trtrs = get_lapack_funcs("trtrs", [B])
    trtrs(B, C, trans=1, lower=0, unitdiag=0, overwrite_b=1)
    trtrs(B, C, trans=1, lower=1, unitdiag=1, overwrite_b=1)

    # C has shape (r, N) -- it is stored transposed
    # find max value in C
    i, j = divmod(abs(C).argmax(), N)

    # set cgeru or zgeru for complex numbers and dger or sger for
    # float numbers
    try:
        ger = get_blas_funcs("geru", [C])
    except:
        ger = get_blas_funcs("ger", [C])

    # set number of iters to 0
    iters = 0
    # check if need to swap rows
    while abs(C[i, j]) > tol and iters < max_iters:

        # add j to index and recompute C by SVM-formula
        index[i] = j
        tmp_row = C[i].copy()
        tmp_column = C[:, j].copy()
        tmp_column[i] -= 1.0
        alpha = -1.0 / C[i, j]
        ger(alpha, tmp_column, tmp_row, a=C, overwrite_a=1)
        iters += 1
        i, j = divmod(abs(C).argmax(), N)

    best_index_state = full_index_set.copy()
    best_index_state = full_index_set[index[:r]]
    return index[:r], best_index_state


# @nb.njit()  # -> This would benefit tremendously froom numba
def greedy_pivot_finder(
    A: np.ndarray,
    I_1i: np.ndarray,
    current_I_pos: list,
    J_1j: np.ndarray,
    current_J_pos: list,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Greedy pivot finder algorithm, which given a matrix A and the current cross aproximations obtained from rows I
    and columns J, finds a new pivot (i_new, j_new) that minimizes the difference between A and Approx.

    Args:
        A (np.ndarray): The input matrix of shape (n, r) which we want to approximate.

        I_1i (np.ndarray): All the rows of A that form the cross approximation, in terms of sets of indices of all the
        k-1 previous sites times all the values for the current site.

        current_I_pos (np.ndarray): The current best rows of A, as in the position of the current best sets of indices
        I in the array of all possible sets of indices I_1i.

        J_1j (np.ndarray): The current best columns of A that form the cross approximation, in terms of sets of indices of all the
        sites from k+2 rightwards, time all the values for the current (site + 1).

        current_J_pos (np.ndarray): The current best columns of A, as in the position of the best sets of indices J
        in the array of all possible sets of indices J_1j.

        tol (float): The tolerance that the algorithm will use to check for convergence of the approximation.

    Returns:
    """

    # OLD IMPLEMENTATION

    # old_is = []
    # for i in I:
    #     for index, i_ext in enumerate(I_1i):
    #         if np.array_equal(i, i_ext):
    #             old_is.append(index)
    #             break

    # old_js = []
    # for j in J:
    #     for index, j_ext in enumerate(J_1j):
    #         if np.array_equal(j, j_ext):
    #             old_js.append(index)
    #             break

    square_core = A[current_I_pos][:, current_J_pos]
    Approx = A[:, current_J_pos] @ np.linalg.inv(square_core) @ A[current_I_pos]

    # This can be optimized to not have to evaluate so many elements
    # TODO THIS ENDS UP FINDING THE PIVOT ALREADY IN THE CROSS -> IN THIS CASE, WE SHOULD NOT INCREASE THE RANK
    i_new, j_new = divmod(np.argmax(np.abs(A - Approx)), A.shape[1])

    if i_new in current_I_pos or j_new in current_J_pos or np.abs(A[i_new, j_new] - Approx[i_new, j_new]) < tol:
        I = I_1i[current_I_pos]
        J = J_1j[current_J_pos]

        return I, J, len(I), len(J), np.sum(np.abs(A - Approx))

    new_I_pos = current_I_pos + [i_new]
    new_J_pos = current_J_pos + [j_new]

    I_new = I_1i[new_I_pos]
    J_new = J_1j[new_J_pos]

    return I_new, J_new, len(I_new), len(J_new), np.sum(np.abs(A - Approx))

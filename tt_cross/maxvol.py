from ast import Index
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

    n, r = A.shape

    if n < r:
        B = A.T.copy()
    else:
        B = A.copy()

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
    # D = np.linalg.solve(C.T, H.T).T

    trtrs = get_lapack_funcs("trtrs", [C])
    trtrs(C, B.T, trans=1, lower=0, unitdiag=0, overwrite_b=1)
    trtrs(C, B.T, trans=1, lower=1, unitdiag=1, overwrite_b=1)

    iter = 1

    # FInd the first swap
    D = B.T

    i, j = divmod(np.argmax(np.abs(D)), r)

    print(D)

    while np.abs(D[i, j]) > tol and iter < max_iter:
        # Perform the swap on the index list
        tmp = index[i]
        index[i] = index[j]
        index[j] = tmp

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


def py_maxvol(A, tol=1.05, max_iters=100, top_k_index=-1):
    """
    Python implementation of 1-volume maximization.

    See Also
    --------
    maxvol
    """
    # some work on parameters
    if tol < 1:
        tol = 1.0
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < r:
        top_k_index = r
    # set auxiliary matrices and get corresponding *GETRF function
    # from lapack
    B = np.copy(A[:top_k_index], order="F")
    C = np.copy(A.T, order="F")
    H, ipiv, info = get_lapack_funcs("getrf", [B])(B, overwrite_a=1)
    # compute pivots from ipiv (result of *GETRF)
    index = np.arange(N, dtype=np.int32)
    for i in range(r):
        tmp = index[i]
        index[i] = index[ipiv[i]]
        index[ipiv[i]] = tmp

    print(index)
    # solve A = CH, H is in LU format
    B = H[:r]
    # It will be much faster to use *TRSM instead of *TRTRS
    trtrs = get_lapack_funcs("trtrs", [B])
    trtrs(B, C, trans=1, lower=0, unitdiag=0, overwrite_b=1)
    trtrs(B, C, trans=1, lower=1, unitdiag=1, overwrite_b=1)
    # C has shape (r, N) -- it is stored transposed
    # find max value in C
    i, j = divmod(abs(C[:, :top_k_index]).argmax(), top_k_index)
    # set cgeru or zgeru for complex numbers and dger or sger for
    # float numbers
    try:
        ger = get_blas_funcs("geru", [C])
    except:
        ger = get_blas_funcs("ger", [C])
    # set number of iters to 0
    iters = 0
    # check if need to swap rows
    print(C.T)
    while abs(C[i, j]) > tol and iters < max_iters:
        # add j to index and recompute C by SVM-formula
        index[i] = j
        tmp_row = C[i].copy()
        tmp_column = C[:, j].copy()
        tmp_column[i] -= 1.0
        alpha = -1.0 / C[i, j]
        ger(alpha, tmp_column, tmp_row, a=C, overwrite_a=1)
        iters += 1
        i, j = divmod(abs(C[:, :top_k_index]).argmax(), top_k_index)
    return index[:r].copy(), C.T


def greedy_pivot_finder(A: np.ndarray, I: np.ndarray, J: np.ndarray, max_iters=100) -> tuple[int]:
    """Greedy pivot finder algorithm, which given a matrix A and the current cross aproximations obtained from rows I
    and columns J, finds a new pivot (i_new, j_new) that maximizes the difference between A and Approx.

    Args:
        A (np.ndarray): The input matrix of shape (n, r) which we want to approximate.
        I (np.ndarray): The current best rows of A that form the cross approximation.
        J (np.ndarray): The current best columns of A that form the cross approximation.
        max_iters (int, optional): The maximum number of updates to the new indices. Defaults to 100.

    Raises:
        ValueError: If I and J are not 1D arrays.
        IndexError: If I or J contain indices that are out of bounds of the input matrix.

    Returns:
        tuple[int]: The new indices (i_new, j_new) that improve the approximation.
    """

    if len(I.shape) != len(J.shape) != 1:
        raise ValueError("I and J must be 1D arrays")
    if any(I > A.shape[0]) or any(J > A.shape[1]):
        raise IndexError("I and J must be within the bounds of the input matrix")

    Approx = A[I][:, J]

    i_new, j_new = divmod(np.argmax(np.abs(A - Approx)), A.shape[1])

    for _ in max_iters:
        i_new = np.argmax(np.abs(A[:, j_new] - Approx[:, j_new]))
        j_new = np.argmax(np.abs(A[i_new] - Approx[i_new]))

        if np.argmax(np.abs(A[:, j_new] - Approx[:, j_new])) <= np.abs(A[i_new, j_new] - Approx[i_new, j_new]) and (
            np.argmax(np.abs(A[i_new] - Approx[i_new])) <= np.argmax(np.abs(A[i_new, j_new] - Approx[i_new, j_new]))
        ):
            return i_new, j_new

    return i_new, j_new

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


def py_maxvol(A: np.ndarray, full_index_set: np.ndarray, tol=1.05, max_iters=100) -> tuple[np.ndarray, np.ndarray]:
    """
    Maxvol algorithm implemented following the original paper:
    Goreinov, Sergei & Oseledets, Ivan & Savostyanov, D. & Tyrtyshnikov, E. & Zamarashkin, Nickolai. (2010).
    "How to Find a Good Submatrix". Matrix Methods: Theory, Algorithms and Applications. 10.1142/9789812836021_0015.

    To improve the performance of the algorithm, the function starts off from the pivots of the LU decomposition of the
    input matrix A. In terms of speed, the algorithm uses the BLAS and LAPACK libraries to perform the operations on the
    matrices. The algorithm will return the indices of the rows of A that form a submatrix with the largest volume, and
    the best set of indices from the original full index set.

    The algorithm follows the steps below:
    - The input matrix will be renamed to B, which will be a copy of A, transposed if n < r.
    - The maximal submatrix will be C, which initially will be asumed to lie in the first r rows of B.
                                  | I |
    - We know that B * C^-1 = D = |   | , where I is an identity of shape (r, r).
                                  | Z |
    - We will solve the system C^T * X = B^T, where X = D^T. This will give us the matrix D.
    - And with D, we can just find max(D[i, j]), add the swap i<->j to the list of indices, and update D with the
    procedure described in the paper, with an update to just the Z part of D.

    The algorithm is adapted from the original implementation in the maxvolpy library https://github.com/c-f-h/maxvolpy/tree/master
    to the needs of the TTRC algorithm implemented in this package.

    Args:
        - A (np.ndarray): The input matrix of shape (n, r). If n < r, the algorithm will transpose the matrix.

        - full_index_set (np.ndarray): The full index set that composes the rows of the input matrix A (the columns in
        case where n < r).

        - tol (float): The tolerance 1+delta that the algorithm will use to check for convergence.

        - max_iters (int): A maximum number of iterations that the algorithm will perform.

    Returns:
        - tuple[np.ndarray, np.ndarray]: The indices of the rows of A that form a submatrix with the largest volume, and
        the best set of indices from the original full index set.
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

    best_index_set = full_index_set.copy()
    best_index_set = full_index_set[index[:r]]
    return index[:r], best_index_set


def greedy_pivot_finder(
    A: np.ndarray,
    I_1i: np.ndarray,
    current_I_pos: list,
    J_1j: np.ndarray,
    current_J_pos: list,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Greedy pivot finder algorithm. Given a matrix A obtained from evalutaing a function in the points
    (I_1i = I_{k-1}⊗i_k, J_1j = J_{k+1}⊗i_{k+1}) the algorithm updates the current cross approximation given by the
    pivots in the sets I and J, which lie in the positions current_I_pos and current_J_pos in the sets I_1i and J_1j.
    To do so, it adds to the approximation the pivot (i_new, j_new) that minimizes the difference between A and the
    approximation, taking advatnage of the interpolation properties of the cross approximation. This property guarantees
    that upon adding a pivot to the cross, the error in the added row and column will be zero. By running the algorithm
    iteratively, we can find the best cross approximation for the given matrix A in a greedy manner.

    Args:
        - A (np.ndarray): The input matrix of shape (n, r) which we want to approximate.

        - I_1i (np.ndarray): All the rows of A that form the cross approximation, as the expanded set I_{k-1}⊗i_k

        - Icurrent_I_pos (np.ndarray): The current best rows of A, as in the position of the current best sets of indices
        I in the array of all possible sets of indices I_1i = I_{k-1}⊗i_k.

        - IJ_1j (np.ndarray): All the columns of A that form the cross approximation, as the expanded set J_{k+1}⊗i_{k+1}.

        - Icurrent_J_pos (np.ndarray): The current best columns of A, as in the position of the best sets of indices J
        in the array of all possible sets of indices J_1j = J_{k+1}⊗i_{k+1}.

        - tol (float): The tolerance to check for convergence. Once the difference between A and the approximation in the
        position (i_new, j_new) is smaller than tol, the algorithm will stop. Setting tol to values below 1e-10 is not
        recommended, as the algorithm may produce singular matrices by selecting pivots that are really really similar
        to what is already in the approximation.

    Returns:
        - tuple[np.ndarray, np.ndarray, int, int, float]: The new sets of indices I and J, the number of rows and columns
        in the new sets, and the error in the approximation, computed as the sum of all the elements in |A - Approx|.
    """
    # Compute first the cross approximation of A with the current sets of indices
    square_core = A[current_I_pos][:, current_J_pos]
    Approx = A[:, current_J_pos] @ np.linalg.inv(square_core) @ A[current_I_pos]

    # Find the pivot that maximizes the difference between A and the approximation
    i_new, j_new = divmod(np.argmax(np.abs(A - Approx)), A.shape[1])

    # If the pivot is already in the current sets of indices, or the difference between A and the approximation in this
    # position is smaller than the tolerance, return the current sets of indices and the error in the approximation
    if i_new in current_I_pos or j_new in current_J_pos or np.abs(A[i_new, j_new] - Approx[i_new, j_new]) < tol:
        I = I_1i[current_I_pos]
        J = J_1j[current_J_pos]

        return I, J, len(I), len(J), np.sum(np.abs(A - Approx))

    # Otherwise, add the pivot to the sets of indices and update the approximation
    new_I_pos = current_I_pos + [i_new]
    new_J_pos = current_J_pos + [j_new]

    I_new = I_1i[new_I_pos]
    J_new = J_1j[new_J_pos]

    return I_new, J_new, len(I_new), len(J_new), np.sum(np.abs(A - Approx))

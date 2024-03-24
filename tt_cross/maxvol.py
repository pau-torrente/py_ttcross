import numpy as np
import numba as nb
from scipy.linalg import get_blas_funcs, get_lapack_funcs


@nb.jit(nopython=True)
def maxvol(A: np.ndarray, complex: bool, tol: float, max_iter: int) -> list[int]:

    if len(A.shape) != 2:
        raise ValueError("A must be a matrix")

    n, r = A.shape

    if n < r:
        B = A.T.copy()
    else:
        B = A.copy()

    C = B[:r, :]

    D_T = np.linalg.solve(C.T, B.T)
    D = D_T.T

    index = np.arange(n)

    iter = 1

    i, j = divmod(np.argmax(np.abs(D)), r)

    while np.abs(D[i, j]) > tol and iter < max_iter:
        index[j] = i

        tmp_row = D[i].copy()
        tmp_column = D[:, j].copy()
        tmp_column[i] -= 1.0
        tmp_row[j] += 1.0
        coeff = -1.0 / D[i, j]

        D[r:] = D[r:] + coeff * np.outer(tmp_column, tmp_row)[r:]
        i, j = divmod(np.argmax(np.abs(D)), r)

        iter += 1

    return index[:r]


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

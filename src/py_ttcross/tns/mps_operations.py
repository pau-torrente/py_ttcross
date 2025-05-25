import numpy as np
from ncon import ncon
import scipy.linalg as la
from copy import deepcopy


def left_orthogonalize_tensor(
    tensor: np.ndarray, dtol: float = 1e-12, left_leg: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Left orthogonalizes a given tensor from an MPS.

    Args:
        - tensor (np.ndarray): The input tensor to be left orthogonalized.
        - dtol (float, optional): The tolerance for determining non-zero eigenvalues. Defaults to 1e-12.
        - left_leg (bool, optional): Whether the tensor from the mps has a left leg in the chain. Used to orthogonalize
        the leftmost tensor in mps without dummy legs at the ends. Defaults to True.
    Returns:
        - np.ndarray: The left orthonormalized tensor.
        - np.ndarray: The matrix used for orthogonalizaion that must be plugged into the tensor to the right of the
        input tensor in the MPS.
    """

    rho = (
        ncon([tensor, np.conj(tensor)], [[1, 2, -1], [1, 2, -2]])
        if left_leg
        else ncon([tensor, np.conj(tensor)], [[1, -1], [1, -2]])
    )

    stemp, utemp = la.eigh(rho)

    # take the non-zero eigenvalues. Keep in mind that la.eigh returns the eigenvalues in increasing order
    chitemp = len(stemp[stemp > dtol])

    d = stemp[-chitemp:]
    u = utemp[:, -chitemp:]

    # the square root of the eigenvalues
    sq_d = np.sqrt(np.abs(d))

    # X is the square root of rho and Xinv is the inverse of X
    x = np.conj(u) @ np.diag(sq_d) @ u.T
    x_inv = np.conj(u) @ np.diag(1 / sq_d) @ u.T

    # we merge Xinv to the tensor such that it becomes leftorthogonal
    newtensor = ncon([tensor, x_inv], [[-1, -2, 1], [1, -3]]) if left_leg else ncon([tensor, x_inv], [[-1, 1], [1, -2]])

    # the matrix X will be plugged to the right neighbor
    return newtensor, x


def right_orthogonalize_tensor(tensor: np.ndarray, dtol=1e-12, right_leg: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Right orthogonalizes a given tensor from an MPS.

    Args:
        - tensor (np.ndarray): The input tensor to be left orthogonalized.
        - dtol (float, optional): The tolerance for determining non-zero eigenvalues. Defaults to 1e-12.
        - right_leg (bool, optional): Whether the tensor from the mps has a right leg in the chain. Used to orthogonalize
        the rightmost tensor in mps without dummy legs at the ends. Defaults to True.

    Returns:
        - np.ndarray: The right orthogonalized tensor.
        - np.ndarray: The matrix used for orthogonalizaion that must be plugged into the tensor to the left of the input tensor in the MPS.
    """
    rho = (
        ncon([tensor, np.conj(tensor)], [[-1, 1, 2], [-2, 1, 2]])
        if right_leg
        else ncon([tensor, np.conj(tensor)], [[-1, 1], [-2, 1]])
    )

    stemp, utemp = la.eigh(rho)

    # take the non-zero eigenvalues. Keep in mind that la.eigh returns the eigenvalues in increasing order
    chitemp = len(stemp[stemp > dtol])

    d = stemp[-chitemp:]
    u = utemp[:, -chitemp:]

    # the square root of the eigenvalues
    sq_d = np.sqrt(np.abs(d))

    # X is the square root of rho and Xinv is the inverse of X
    x = np.conj(u) @ np.diag(sq_d) @ u.T
    x_inv = np.conj(u) @ np.diag(1 / sq_d) @ u.T

    # we merge Xinv to the tensor such that it becomes rightorthogonal
    newtensor = (
        ncon([x_inv, tensor], [[-1, 1], [1, -2, -3]]) if right_leg else ncon([x_inv, tensor], [[-1, 1], [1, -2]])
    )

    # the matrix X will be plugged to the left neighbor
    return newtensor, x


class OrthoOps:
    @staticmethod
    def to_left_orthogonal(
        mps: np.ndarray,
        dummy_ends: bool = True,
        site: int = None,
        dtol: float = 1e-12,
        get_matrices: bool = False,
        normalize: bool = False,
    ):
        """
        Convert the first `site` tensors of the MPS to left-orthogonal form. If the get_matrices flag is set to True,
        the matrices used to make the MPS left-orthogonal at each site are also returned.

        Args:
            mps (np.ndarray): The array of tensors representing an MPS to be made left orthogonal.
            dummy_ends (bool, optional): Whether the MPS has dummy legs at the ends. Defaults to True.
            site (int): The number of sites of the MPS to be made left-orthogonal. If a site is not given, the entire
            MPS is made left-orthogonal. Defaults to None.
            get_matrices (bool, optional): Whether to return the matrices used to make the MPS left-orthogonal at each
            site. Defaults to False.
            normalize (bool, optional): Whether to normalize the MPS taking advantag of the orthogonality property
            introducied in the tensor network. Defaults to False.
        """
        weight_list = []
        tensors = mps.copy()

        if site is None:
            nstop = len(mps) - 1
        elif isinstance(site, int) and site < len(mps):
            nstop = site
        else:
            raise ValueError("Given sites must be an integer less than the number of sites minus one. ")

        for i in range(nstop):
            if i == 0 and dummy_ends:
                tensors[i], s_matrix = left_orthogonalize_tensor(tensors[i], dtol, left_leg=False)
            else:
                tensors[i], s_matrix = left_orthogonalize_tensor(tensors[i], dtol, left_leg=True)

            weight_list.append(s_matrix) if get_matrices else None
            tensors[i + 1] = ncon([s_matrix, tensors[i + 1]], [[-1, 1], [1, -2, -3]])

        if normalize:
            norm = (
                ncon([tensors[-1], np.conj(tensors[1])], [[1, 2, 3], [1, 2, 3]])
                if dummy_ends
                else ncon([tensors[-1], np.conj(tensors[1])], [[1, 2], [1, 2]])
            )

            tensors[-1] /= np.sqrt(norm)

        return tensors, weight_list if get_matrices else tensors

    @staticmethod
    def to_right_orthogonal(
        mps: np.ndarray,
        dummy_ends: bool = True,
        site: int = None,
        dtol: float = 1e-12,
        get_matrices: bool = False,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Convert the last `site` tensors of the MPS to right-orthogonal form. If the get_matrices flag is set to True,
        the matrices used to make the MPS left-orthogonal at each site are also returned.

        Args:
            mps (MPS): The MPS to be made right-orthogonal.
            dummy_ends (bool, optional): Whether the MPS has dummy legs at the ends. Defaults to True.
            site (int): The number of sites of the MPS, starting from the right, to be made right-orthogonal. If a site
            is not given, the entire MPS is made right-orthogonal. Defaults to None.
            get_matrices (bool, optional): Whether to return the matrices used to make the MPS right-orthogonal at each
            site. Defaults to False.
            normalize (bool, optional): Whether to normalize the MPS taking advantage of the orthogonality property
            introducied in the tensor network. Defaults to False.
        """

        weight_list = []
        tensors = mps.copy()

        if site is None:
            nstop = len(mps)
        elif isinstance(site, int) and site < len(mps):
            nstop = site + 1
        else:
            raise ValueError("Given sites must be an integer less than the number of sites. ")

        for i in range(-1, -nstop, -1):
            if i == -1 and not dummy_ends:
                tensors[i], s_matrix = right_orthogonalize_tensor(tensors[i], dtol, right_leg=False)
            else:
                tensors[i], s_matrix = right_orthogonalize_tensor(tensors[i], dtol, right_leg=True)

            weight_list.append(s_matrix) if get_matrices else None
            tensors[i - 1] = (
                ncon([tensors[i - 1], s_matrix], [[-1, 1], [1, -2]])
                if i == -nstop + 1 and not dummy_ends
                else ncon([tensors[i - 1], s_matrix], [[-1, -2, 1], [1, -3]])
            )

        if normalize:
            norm = (
                ncon([tensors[0], np.conj(tensors[1])], [[1, 2, 3], [1, 2, 3]])
                if dummy_ends
                else ncon([tensors[0], np.conj(tensors[1])], [[1, 2], [1, 2]])
            )

            tensors[0] /= np.sqrt(norm)

        return tensors, weight_list[::-1] if get_matrices else tensors

def mps_to_mpo(mps: np.ndarray):
    """
    Converts an MPS representing a function into an MPO that has the entries of the function in the diagonals. Uses the copy tensor ∂_{a, b, g} = 1 if a = b = g to split the 
    physical index of each of the tensors of the MPS into two.

    Args:
        - mps (np.ndarray): Tensors that form the MPS

    Returns:
        - np.ndarray: Tensors of the MPO with the entries of the MPS in the diagonals of the physical indices.
    """
    copy_tensor = np.zeros((2,2,2))
    copy_tensor[0, 0, 0] = 1.0
    copy_tensor[1, 1, 1] = 1.0

    mpo = deepcopy(mps)
    for tensor in mpo:
        tensor = ncon([tensor, copy_tensor], [[-1, 1, -4], [1, -2, -3]])
    return mpo
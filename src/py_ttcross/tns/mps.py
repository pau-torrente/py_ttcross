
import numpy as np

def _overflow_checker(tensors: np.ndarray, physical_indices: list[int], site: int, max_bond_dimension: int) -> int:
    """Helper function that avoids computing extremely large numbers that overflow when computing the shape of
    of tensors in the MPS. When computing the shapes, this function substitues computing a very large power of
    the phyisical dimension, by doing the multiplications one by one and checking if the result is larger than the
    maximum bond dimension. If it is, it returns the current minimum value that the bond dimension can take without
    overflowing.

    Args:
        site (int): The site of the MPS where we are computing the shape of the tensor.
        max_bond_dimension (int): The maximum bond dimension that the tensor can have.

    Returns:
        int: The value of the bond dimension.
    """
    current_min = min(max_bond_dimension, tensors[site - 1].shape[-1] * physical_indices[site])
    prod = 1
    for i in range(site + 1, len(physical_indices)):
        prod *= physical_indices[i]
        if prod > current_min:
            return current_min
    return prod

def _tensor_shapes(tensors: np.ndarray, physical_indices: list[int], site: int, max_bond_dimension: int) -> tuple[int, int, int]:
    """Helper method that computes the shapes that the tensors of the MPS must have at each site. It takes into
    account the exponential scaling of the bond dimension from the ends to the center of the MPS, while capping the
    bond dimensions to the maximum predefined bond dimension.

    Args:
        site (int): The site of the MPS where we are computing the shape of the tensor.
        max_bond_dimension (int): The maximum bond dimension that the tensor can have.

    Returns:
        tuple[int, int, int]: The shape of the tensor at the given site.
    """


    if site == 0:
        return physical_indices[0], int(min(physical_indices[0], max_bond_dimension))

    elif site == len(physical_indices) - 1:
        return int(min(physical_indices[-1], max_bond_dimension)), physical_indices[-1]

    else:
        return (
            tensors[site - 1].shape[-1],
            physical_indices[site],
            _overflow_checker(tensors, physical_indices, site, max_bond_dimension),
        )

def create_random_mps(
    n_sites: int,
    max_chi: int,
    complex_entries: bool,
) -> np.ndarray:
    """Method that creates an MPS with random entries and phys_d = 2. The entries can be real or complex.

    Args:
        n_sites (int): The number of sites of the MPS.
        phys_d (int | list[int]): The dimension of the physical legs of the MPS. It can be an int that sets all the
            legs to the same dimension, or a list of ints, that gives a unique dimension to each leg.
        max_chi (int): The maximum bond dimension that the tensors of the MPS can have.
        complex_entries (bool): Whether the entries of the tensors are real or complex.

    Returns:
        np.ndarray: An array of tensors representing an MPS object with random entries.
    """

    tensors = np.ndarray(n_sites, dtype=object)
    phys_d = [2] * n_sites

    for i in range(n_sites):
        tensors[i] = (
            np.random.rand(*_tensor_shapes(tensors, phys_d, i, max_chi)) + 1j * np.random.rand(*_tensor_shapes(tensors, phys_d, i, max_chi))
            if complex_entries
            else np.random.rand(*_tensor_shapes(tensors, phys_d, i, max_chi))
        )
    # for i in range(n_sites):
    #     tensors[i] = (
    #         np.ones(_tensor_shapes(tensors, phys_d, i, max_chi)) + 1j * np.ones(_tensor_shapes(tensors, phys_d, i, max_chi))
    #         if complex_entries
    #         else np.ones(_tensor_shapes(tensors, phys_d, i, max_chi))
    #     )
    return tensors


from abc import ABC, abstractmethod
import numpy as np
from ncon import ncon

class DifferentialMPO:
    def __init__(self, L: int):
        self.L = L
        self.build_operator()
    
    @abstractmethod
    def build_operator():
        pass


class OneDimLaplacian(DifferentialMPO):
    # Leg ordering: leftbond-upphys-lowphys-rightbond
    def __init__(self, L: int):
        super().__init__(L)

    def _build_leftmost_tensor(self):
        left = np.zeros(3)
        left[0] = 1.0

        second = self._build_inner_tensor()
        return ncon([left, second], [[1], [1, -1, -2, -3]])
    
    def _build_inner_tensor(self):
        tens = np.zeros((3,2,2,3))
        tens[0, 0, 0, 0] = 4.0
        tens[0, 1, 1, 0] = 4.0
        tens[1, 1, 0, 1] = 4.0
        tens[0, 0, 1, 1] = 4.0
        tens[0, 1, 0, 2] = 4.0
        tens[2, 0, 1, 2] = 4.0
        return tens
    
    def _build_rightmost_tensor(self):
        right = np.zeros(3)
        right[0] = -2.0
        right[1] = 1
        right[2] = 1

        second_to_last = self._build_inner_tensor()

        return ncon([second_to_last, right], [[-1, -2, -3, 1], [1]])

    def build_operator(self):
        self.laplacian = np.ndarray(self.L, dtype=np.ndarray)
        self.laplacian[0] = self._build_leftmost_tensor()
        for k in range(1, self.L-1):
            self.laplacian[k] = self._build_inner_tensor()
        self.laplacian[self.L - 1] = self._build_rightmost_tensor()
        return self.laplacian
    
class OneDimHeatEqEvolver(DifferentialMPO):
    def __init__(self, L: int, dt: float):
        super().__init__(L)
        self.laplacian = OneDimLaplacian(L)
        self.dt = dt
        self.id = np.eye(2)

    def _build_leftmost_tensor(self):
        big_block = self.dt * self.laplacian[0]
        laplacian_bond_right = self.laplacian[0].shape[2]

        final_tensor_shape = big_block.shape + np.array([0, 0, 1])
        final_tensor = np.zeros(final_tensor_shape, dtype = big_block.dtype)

        final_tensor[:, :, :laplacian_bond_right] = big_block
        final_tensor[:, :, laplacian_bond_right] = self.id

        return final_tensor
    
    def _build_rightmost_tensor(self):
        big_block = self.dt * self.laplacian[-1]
        laplacian_bond_left = self.laplacian[-1].shape[0]

        final_tensor_shape = big_block.shape + np.array([1, 0, 0])
        final_tensor = np.zeros(final_tensor_shape, dtype = big_block.dtype)

        final_tensor[:laplacian_bond_left, :, :] = big_block
        final_tensor[laplacian_bond_left, :, :] = self.id

        return final_tensor
    
    def _build_inner_tensor(self, site:int):
        big_block = self.dt * self.laplacian[site]
        laplacian_bond_left = self.laplacian[site].shape[0]
        laplacian_bond_right = self.laplacian[site].shape[3]

        final_tensor_shape = big_block.shape + np.array([1, 0, 0, 1])
        final_tensor = np.zeros(final_tensor_shape, dtype = big_block.dtype)

        final_tensor[:laplacian_bond_left, :, :, :laplacian_bond_right] = big_block
        final_tensor[laplacian_bond_left, :, :, laplacian_bond_right] = self.id

        return final_tensor
    
    def build_operator(self):
        self.heat_eq_evolver = np.ndarray(self.L, dtype=np.ndarray)
        self.heat_eq_evolver[0] = self._build_leftmost_tensor()
        for k in range(1, self.L - 1):
            self.heat_eq_evolver[k] = self._build_inner_tensor()
        self.heat_eq_evolver[self.L - 1] = self._build_rightmost_tensor()
        return self.heat_eq_evolver
    

class Prolongation:
    # Leg ordering: leftbond-upphys-lowphys-rightbond
    def __init__(self, L:int):
        self.L = L
        self.build_operator()

    def _build_leftmost_tensor(self):
        tensor = self._build_innner_tensor()
        return tensor[0, :, :, :]
    
    def _build_innner_tensor(self):
        tens = np.zeros((2,2,2,2))
        tens[0, 0, 0, 0] = 1.0
        tens[0, 1, 1, 0] = 1.0
        tens[0, 1, 0, 1] = 1.0
        tens[1, 0, 1, 1] = 1.0
        return tens
    
    def _build_rightmost_tensor(self):
        tens = np.zeros((2,2))
        tens[0, 0] = 1.0
        tens[0, 1] = 0.5
        tens[1, 1] = 0.5
        return tens

    def build_operator(self):
        self.prolong_mpo = np.ndarray(self.L + 1, dtype=np.ndarray)
        self.prolong_mpo[0] = self._build_leftmost_tensor()

        for k in range(1, self.L):
            self.prolong_mpo[k] = self._build_innner_tensor()

        self.prolong_mpo[self.L] = self._build_rightmost_tensor()
        return self.prolong_mpo
        

        






    
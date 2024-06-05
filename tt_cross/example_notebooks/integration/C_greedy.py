import sys

sys.path.append("/home/ptbadia/code/tfg/tfg_ttcross")

from tt_cross.src.regular_tt_cross.integrators import tracked_greedycross_integrator
import matplotlib.pyplot as plt
import numpy as np
from tt_cross.src.utils.functions import *
import csv


def test_function(x: np.ndarray) -> np.ndarray:
    return 2 / (
        (1 + np.sum([np.prod(x[: k + 1]) for k in range(len(x))]))
        * (1 + np.sum([np.prod(x[k:]) for k in range(len(x))]))
    )


def write_to_csv(filename: str, data: list):
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


class csv_tracked_greedycross(tracked_greedycross_integrator):
    def index_update(self, site: int):
        super().index_update(site)

        # print(self.evolution)
        # print("=========")
        write_to_csv(
            "/home/ptbadia/code/tfg/tfg_ttcross/tt_cross/example_notebooks/integration/C16_data/C16_greedy_100_numba.csv",
            self.evolution[-1],
        )

    def full_sweep(self) -> None:
        """Perform a full left to right and right to left sweep in the tensor train."""
        for site in range(self.num_variables - 1):
            self.index_update(site)

        for site in range(self.num_variables - 2, -1, -1):
            self.index_update(site)

        print("Time spent on superblocks:", self.super_block_time)
        print("Time spent in total:", self.total_time)


integrator2 = csv_tracked_greedycross(
    func=test_function,
    num_variables=64,
    intervals=np.array([[0, 1] for _ in range(64)]),
    points_per_variable=100,
    sweeps=12,
    max_bond=24,
    quadrature="Gauss",
    pivot_finder_tol=1e-6,
)

integrator2.integrate()

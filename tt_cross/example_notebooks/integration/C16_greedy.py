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

        write_to_csv(
            "/home/ptbadia/code/tfg/tfg_ttcross/tt_cross/example_notebooks/integration/C16_data/C16_greedy_100.csv",
            self.evolution[-1],
        )


integrator2 = csv_tracked_greedycross(
    func=test_function,
    num_variables=16,
    intervals=np.array([[0, 1] for _ in range(16)]),
    points_per_variable=100,
    sweeps=10,
    max_bond=20,
    quadrature="Gauss",
    pivot_finder_tol=1e-6,
)

integrator2.integrate()

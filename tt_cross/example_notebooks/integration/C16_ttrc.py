import sys

sys.path.append("/home/ptbadia/code/tfg/tfg_ttcross")

from tt_cross.src.regular_tt_cross.integrators import tracked_ttrc_integrator
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


class csv_tracked_ttrc(tracked_ttrc_integrator):

    def full_sweep(self) -> None:
        super().full_sweep()

        write_to_csv(
            "/home/ptbadia/code/tfg/tfg_ttcross/tt_cross/example_notebooks/integration/C16_ttrc2.csv",
            self.evolution[-1],
        )


integrator = csv_tracked_ttrc(
    func=test_function,
    num_variables=16,
    intervals=np.array([[0, 1] for _ in range(16)]),
    points_per_variable=60,
    sweeps=6,
    initial_bond_guess=5,
    max_bond=20,
    quadrature="Trapezoidal",
    truncation_tol=1e-10,
    maxvol_tol=1e-5,
)
integrator.integrate()

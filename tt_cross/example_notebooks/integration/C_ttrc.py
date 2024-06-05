import sys

sys.path.append("/home/ptbadia/code/tfg/tfg_ttcross")

from tt_cross.src.regular_tt_cross.integrators import tracked_ttrc_integrator
from tt_cross.src.quantic_tt_cross.integrators import ttrc_qtt_integrator
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


for bond in range(5, 15):

    integrator = tracked_ttrc_integrator(
        func=test_function,
        num_variables=64,
        intervals=np.array([[0, 1] for _ in range(64)]),
        points_per_variable=100,
        sweeps=3,
        initial_bond_guess=3,
        max_bond=bond,
        quadrature="Gauss",
        truncation_tol=1e-12,
        maxvol_tol=1e-8,
    )
    integrator.integrate()

    print(integrator.bonds)
    print("Time spent on superblocks:", integrator.super_block_time)
    print("Time spent in total:", integrator.total_time)

    write_to_csv(
        "/home/ptbadia/code/tfg/tfg_ttcross/tt_cross/example_notebooks/integration/C64_data/C64_ttrc_numba_100.csv",
        integrator.evolution[-1],
    )

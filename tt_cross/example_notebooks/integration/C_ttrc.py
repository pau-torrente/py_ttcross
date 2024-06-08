import sys

sys.path.append("/home/ptbadia/code/tfg/tfg_ttcross")

from tt_cross.src.regular_tt_cross.integrators import tracked_ttrc_integrator, ttrc_integrator
from tt_cross.src.quantic_tt_cross.integrators import ttrc_qtt_integrator
import matplotlib.pyplot as plt
import numpy as np
from tt_cross.src.utils.functions import *
import csv


C16_exact = 0.6305039461732372635052956575606874194843162172081030477508791197370587113428
518776591927635011910666019821885772282625005863790302590212510471642111230055
846525034440766011716943063675091961344295295167762531039303033076338954225849
425176347989010624576159605228245752442523276560004610937432747935264686038248
528719167665214134983765365722519250395916835193811814313121457043515985621220
385335330522425818627568844202427436280607422676722152074638421633970966585698
33805864256285865788069


def test_function(x: np.ndarray) -> np.float64:
    t1 = 0
    for k in range(x.shape[0]):
        t1 += np.prod(x[: k + 1])

    t2 = 0
    for k in range(x.shape[0]):
        t2 += np.prod(x[k:])

    return 2.0 / ((1.0 + t1) * (1.0 + t2))


def write_to_csv(filename: str, data: list):
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


initial_b_guess = [1, 2, 3, 4, 5, 4, 4, 4]
sw = [3, 4, 4, 5, 6]
for i, bond in enumerate(range(1, 6)):
    integrator = ttrc_integrator(
        func=test_function,
        num_variables=63,
        intervals=np.array([[0, 1] for _ in range(63)]),
        points_per_variable=5,
        sweeps=sw[i],
        initial_bond_guess=initial_b_guess[i],
        max_bond=bond,
        quadrature="Gauss",
        truncation_tol=1e-6,
        maxvol_tol=1e-12,
    )
    val = integrator.integrate()

    print(integrator.interpolator.bonds)
    print("Time spent on superblocks:", integrator.interpolator.super_block_time)
    print("Time spent in total:", integrator.interpolator.total_time)

    write_to_csv(
        "/home/ptbadia/code/tfg/tfg_ttcross/tt_cross/example_notebooks/integration/C64_data/C64_ttrc_numba_5_.csv",
        [integrator.interpolator.func_calls, val],
    )
    # for e in integrator.evolution:

    #     write_to_csv(
    #         "/home/ptbadia/code/tfg/tfg_ttcross/tt_cross/example_notebooks/integration/C64_data/C64_ttrc_numba_5_2.csv",
    #         e,
    #     )

# integrator = tracked_ttrc_integrator(
#     func=test_function,
#     num_variables=63,
#     intervals=np.array([[0, 1] for _ in range(63)]),
#     points_per_variable=ppv_list[i],
#     sweeps=swep_list[i],
#     initial_bond_guess=init_bond_list[i],
#     max_bond=bond,
#     quadrature="Gauss",
#     truncation_tol=1e-8,
#     maxvol_tol=1e-12,
# )
# integrator.integrate()

# print(integrator.bonds)
# print("Time spent on superblocks:", integrator.super_block_time)
# print("Time spent in total:", integrator.total_time)

# write_to_csv(
#     "/home/ptbadia/code/tfg/tfg_ttcross/tt_cross/example_notebooks/integration/C64_data/C64_ttrc_numba_5_.csv",
#     integrator.evolution[-1],
# )

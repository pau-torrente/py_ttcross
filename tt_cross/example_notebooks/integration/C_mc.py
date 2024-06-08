import numba
import numpy as np
import csv
import pandas as pd


@numba.jit(nopython=True)
def test_function(x: np.ndarray) -> np.float64:
    t1 = 0
    for k in range(x.shape[0]):
        t1 += np.prod(x[: k + 1])

    t2 = 0
    for k in range(x.shape[0]):
        t2 += np.prod(x[k:])

    return 2.0 / ((1.0 + t1) * (1.0 + t2))


@numba.jit(nopython=True)
def c_with_MC(n: int, n_samples: int) -> np.float64:

    grid = np.empty((n, 60), dtype=np.float64)
    for i in range(n):
        grid[i, :] = np.linspace(0, 1, 60)

    integral = []
    summ = 0
    for i in range(n_samples):
        print(i) if i % 1000000 == 0 else None

        point = grid[np.arange(grid.shape[0]), np.random.choice(grid.shape[1], grid.shape[0])]

        summ += test_function(point)

        integral.append([i + 1, summ / (i + 1)])

    return np.array(integral, dtype=np.float64)


@numba.jit(nopython=True)
def c_with_MC(n: int, n_samples: int) -> np.float64:

    # grid = np.empty((n, 60), dtype=np.float64)
    # for i in range(n):
    #     grid[i, :] = np.linspace(0, 1, 60)

    integral = np.empty((n_samples, 2), dtype=np.float64)
    summ = 0
    for i in range(n_samples):
        print(i) if i % 1000000 == 0 else None

        point = np.random.rand(n)

        summ += test_function(point)

        integral[i][0] = i + 1
        integral[i][1] = summ / (i + 1)

    return integral


def write_to_csv(filename: str, data: list):
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


integral = c_with_MC(63, 5000000)

dF = pd.DataFrame(integral[::10], columns=["n_samples", "integral"])

dF.to_csv("/home/ptbadia/code/tfg/tfg_ttcross/tt_cross/example_notebooks/integration/C64_data/C64_mc_numba4.csv")

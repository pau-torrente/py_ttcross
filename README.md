# py_ttcross
A lightweight python toolkit for interpolation and integration of functions, based on the Tensor Train Cross (TT-Cross) decomposition of large tensors [1]. The package builds the approximation of a black box N-dimensional function using either the TTRC algorithm [2] or the Greedy-Cross algorithm [3]. You can find more information in the references or in my Bachelor's thesis for which I built this toolkit (you can find it in the main folder of this repo as TFG-Pau-Torrente-Badia.pdf)

## Installation
To install the toolkit, clone the respository into the desirded folder in your machine, through https, for example:
```bash
git clone https://github.com/pau-torrente/py_ttcross.git
```
And install the dependencies needed:
```
pip install -r requirements.txt
```

## Usage
### Multi-dimensional integration/interpolation
In order to use the multidimensional interpolation/integration tools of the package, you must start with a function such as:
```python
import numpy as np

def test_function(x: np.ndarray) -> np.float64:
    return np.sinc(np.sum(x))
```
- The function should take in a numpy vector and output a scalar value (it can be a real or complex number)
- The package utilizes numba jit compilation to speed up the building of large tensors. Before giving the function to the interpolation/integration methods, test that it can indeed be compiled in "nopython" mode.

Once you have this function, you can create a grid as:
```python
import numpy as np
num_var = 40
subdivision = 25
grid = np.array([np.linspace(1, 2, subdivision) for _ in range(num_var)])
```
And interpolate the function on top of it, with the ttrc algorithm for example:
```python
from py_ttcross.regular_tt_cross.dmrg_cross import ttrc
ttrc_interpolator = ttrc(
    func=test_function,
    num_variables=num_var,
    grid=grid,
    maxvol_tol=1e-10,
    truncation_tol=1e-10,
    sweeps=5,
    initial_bond_guess=10,
    max_bond=20,
)
test_interpolator.run()
```
Or directly compute its integral using quadrature rules, Gauss-Legendre in this case:
```python
from py_ttcross.regular_tt_cross.integrators import ttrc_integrator
integrator = ttrc_integrator(
    func=test_function,
    num_variables=8,
    intervals=np.array([[0, 1] for _ in range(8)]),
    points_per_variable=10,
    sweeps=4,
    initial_bond_guess=2,
    max_bond=8,
    quadrature="Gauss",
    truncation_tol=0,
    maxvol_tol=1e-8,
)

integrator.integrate
```
In this case, the grid is created internally from the given integrals and the nu,ber of point per variable specified.

### Quantic Tensor Train interpolation/integration
A single variable $x_i=\frac{L}{2^d}\left(i+\frac{1}{2}\right)$, $i = 0, \dots, 2^d-1$ can also be converted into d binary ones [2]:

$i \leftrightarrow (i_1,\dots,i_d), i = \sum_{p = 1}^{d} i_p 2^{p-1}, i_p = 0, 1$

In order to, for example, interpolate one dimensional functions. To do so, you can create a function that takes in a float and outputs another scalar, such as:

```python
import numpy as np
def test_function(x:float) -> np.float_:
    return np.abs(10 * np.sin(np.lox(x+1)))*np.exp(-x)
```
And pass it to the 1D interpolators of the package:
```python
from py_ttcross.quantic_tt_cross.interpolators_1D import ttrc_one_dim_func_interpolator
test_interpolator = ttrc_one_dim_func_interpolator(
    func=test_func,
    interval=interval,
    d=15,
    complex_function=False,
    initial_bond_guess=2,
    max_bond=16,
    maxvol_tol=1e-10,
    truncation_tol=1e-10,
    sweeps=5,
    pivot_initialization="first_n"
)
test_interpolator.interpolate()
```
**More examples can be found in the [example_notebooks](https://github.com/pau-torrente/py_ttcross/tree/main/example_notebooks) of the repo.**

## Troubleshooting
Since this package was a first approach to the TT-Cross framework, not much focus was put on precision and performance. For this reason, very large interpolation instances can be slow and errors might propagate quickly due to the lack of precision (a more in depth discussion of this last point can be found in [1]). If you find that your tests output results which not make much sense, or feel like the algorithms are not able to converge, **try reducing the maximum bond dimension or the tolerances**. This will result in smaller tensors which will, hopefully, not accumulate as much error and be handled better by numpy.



___

### References

[1]: I. Oseledets and E. Tyrtyshnikov, “Tt-cross approximation for multidimensional arrays,” Linear Algebra and its Applications, vol. 432, no. 1, pp. 70–88, 2010.

[2]: D. Savostyanov and I. Oseledets, “Fast adaptive interpolation of multi-dimensional arrays in tensor train format,” pp. 1–8, 2011.

[3]: S. Dolgov and D. Savostyanov, “Parallel cross interpo-lation for high-precision calculation of high-dimensionalvintegrals,” Computer Physics Communications, vol. 246, p. 106869, 2020.

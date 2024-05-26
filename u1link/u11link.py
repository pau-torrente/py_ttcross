import numpy as np
from scipy.special import ivp as modbes1
from scipy.integrate import quad

from types import FunctionType


class ObservablesU11:
    @staticmethod
    def polyakov_loop(x: float, **kwargs):
        return np.exp(1j * x)

    @staticmethod
    def exp_polyakov_loop(beta: float, k: float, mu: complex):
        z = U11linkModel.exact_z(beta, k, mu)
        return (
            modbes1(1, beta, 0) + k * np.cosh(mu) * modbes1(1, beta, 1) - k * modbes1(1, beta, 0) * np.sinh(mu) / beta
        ) / z

    @staticmethod
    def inverse_polyakov_loop(x: float, **kwargs):
        return ObservablesU11.polyakov_loop(-x, **kwargs)

    @staticmethod
    def exp_inverse_polyakov_loop(beta: float, k: float, mu: complex):
        return ObservablesU11.exp_polyakov_loop(beta, k, -mu)

    @staticmethod
    def plaquette(x: float, **kwargs):
        return np.cos(x)

    @staticmethod
    def exp_plaquette(beta: float, k: float, mu: complex):
        z = U11linkModel.exact_z(beta, k, mu)
        return (modbes1(1, beta, 0) + k * np.cosh(mu) * modbes1(1, beta, 1)) / z

    @staticmethod
    def density(x: float, **kwargs):
        k = kwargs.get("k", 0.0)
        mu = kwargs.get("mu", 0.0)

        return (1j * k * np.sin(x - 1j * mu)) / (1 + k * np.cos(x - 1j * mu))

    @staticmethod
    def exp_density(beta: float, k: float, mu: complex):
        z = U11linkModel.exact_z(beta, k, mu)
        return (k * np.sinh(mu) * modbes1(1, beta, 0)) / z


class U11linkModel:
    def bosonic_action(self, beta: float, x: float):
        return -beta * np.cos(x)

    def fermionic_det(self, k: float, mu: complex, x: float):
        return 1 + k * np.cos(x - 1j * mu)

    @staticmethod
    def exact_z(beta: float, k: float, mu: complex):
        return modbes1(0, beta, 0) + k * np.cosh(mu) * modbes1(1, beta, 0)

    def expectation_weight(self, x: float, beta: float, k: float, mu: complex, operator: FunctionType):
        return np.exp(-self.bosonic_action(beta, x)) * self.fermionic_det(k, mu, x) * operator(x, k=k, mu=mu, beta=beta)

    def _one(self, x: float, **kwargs):
        return 1

    def expectation_value(self, beta: float, k: float, mu: complex, operator: FunctionType):
        normalization_weight = lambda x: self.expectation_weight(x, beta, k, mu, self._one)
        normalization = quad(normalization_weight, -np.pi, np.pi, complex_func=True)[0]

        obs_weight = lambda x: self.expectation_weight(x, beta, k, mu, operator)
        return quad(obs_weight, -np.pi, np.pi, complex_func=True)[0] / normalization


class ObservablesU1N:
    @staticmethod
    def polyakov_loop(x: float, **kwargs):
        return np.exp(1j * x)

    @staticmethod
    def exp_polyakov_loop(beta: float, k: float, mu: complex):
        z = U11linkModel.exact_z(beta, k, mu)
        return (
            modbes1(1, beta, 0) + k * np.cosh(mu) * modbes1(1, beta, 1) - k * modbes1(1, beta, 0) * np.sinh(mu) / beta
        ) / z

    @staticmethod
    def inverse_polyakov_loop(x: float, **kwargs):
        return ObservablesU11.polyakov_loop(-x, **kwargs)

    @staticmethod
    def exp_inverse_polyakov_loop(beta: float, k: float, mu: complex):
        return ObservablesU11.exp_polyakov_loop(beta, k, -mu)

    @staticmethod
    def plaquette(x: float, **kwargs):
        return np.cos(x)

    @staticmethod
    def exp_plaquette(beta: float, k: float, mu: complex):
        z = U11linkModel.exact_z(beta, k, mu)
        return (modbes1(1, beta, 0) + k * np.cosh(mu) * modbes1(1, beta, 1)) / z

    @staticmethod
    def density(x: float, **kwargs):
        k = kwargs.get("k", 0.0)
        mu = kwargs.get("mu", 0.0)

        return (1j * k * np.sin(x - 1j * mu)) / (1 + k * np.cos(x - 1j * mu))

    @staticmethod
    def exp_density(beta: float, k: float, mu: complex):
        z = U11linkModel.exact_z(beta, k, mu)
        return (k * np.sinh(mu) * modbes1(1, beta, 0)) / z


class U1NlinkModel:
    def bosonic_action(self, beta: float, x: float):
        return -beta * np.cos(x)

    def fermionic_det(self, k: float, mu: complex, x: float):
        return 1 + k * np.cos(x - 1j * mu)

    @staticmethod
    def exact_z(N: int, beta: float, k: float, mu: complex):
        return (modbes1(0, beta, 0) + k * np.cosh(mu) * modbes1(1, beta, 0)) ** N

    def expectation_weight(self, x: np.ndarray, beta: float, k: float, mu: complex, operator: FunctionType):
        return (
            np.prod(np.exp(-self.bosonic_action(beta, x)))
            * np.prod(self.fermionic_det(k, mu, x))
            * np.sum(operator(x, k=k, mu=mu, beta=beta))
        )

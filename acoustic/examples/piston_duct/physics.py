"""Independent linear acoustic FEM coupled to a cubic-spring piston.

Periodic snapshot synthesis is internal to data generation. No continuation or
stability-analysis interface is provided.
"""
import numpy as np
from numba import njit
from scipy.linalg import solve_banded
from scipy.optimize import root


def parameters(config):
    p = config["physical"]
    return np.array([config["elements"], p["k"], p["c"], p["alpha_scaled"],
                     p["xs"], p["force"], p["mu"], p["eta"], p["gamma"]])


def mass_weights(n):
    w = np.ones(n + 1) / n
    w[[0, -1]] *= .5
    return w


@njit(cache=True)
def fem_rhs(y, force, par):
    n = int(par[0])
    k, c, alpha, xs, _, mu, eta, gamma = par[1:]
    off = n + 3
    out = np.empty_like(y)
    acc = (force / xs - c * y[1] - k * y[0] - alpha * y[0] ** 3
           - mu / xs * (y[2] + eta * y[off]))
    out[0] = y[1]
    out[1] = acc
    for i in range(n + 1):
        h = y[2 + i] + eta * y[off + i]
        out[2 + i] = y[off + i]
        if i == 0:
            lap = 2 * n * n * (y[3] + eta * y[off + 1] - h)
            lap += 2 * n * xs * (acc + gamma * y[1])
        elif i == n:
            lap = 2 * n * n * (y[n + 1] + eta * y[off + n - 1] - h)
        else:
            lap = n * n * (y[i + 1] + eta * y[off + i - 1]
                           - 2 * h + y[i + 3] + eta * y[off + i + 1])
        out[off + i] = lap - gamma * y[off + i]
    return out


@njit(cache=True)
def sample_fem(initial, omega, force_amplitude, par, cycles, spp, per):
    """RK4 integration, record phase zero and per samples/cycle, no endpoint."""
    y = initial.copy()
    dt = 2 * np.pi / (omega * spp)
    states = np.empty((cycles * per, len(y)), np.float32)
    stride = spp // per
    row = 0
    for cy in range(cycles):
        for j in range(spp):
            phase = 2 * np.pi * j / spp
            if j % stride == 0:
                states[row] = y
                row += 1
            k1 = fem_rhs(y, force_amplitude * np.cos(phase), par)
            k2 = fem_rhs(y + .5 * dt * k1,
                         force_amplitude * np.cos(phase + .5 * omega * dt), par)
            k3 = fem_rhs(y + .5 * dt * k2,
                         force_amplitude * np.cos(phase + .5 * omega * dt), par)
            k4 = fem_rhs(y + dt * k3,
                         force_amplitude * np.cos(phase + omega * dt), par)
            y += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return states, y


class PeriodicSnapshots:
    """Fixed-frequency harmonic equations used ONLY to synthesize snapshots."""
    def __init__(self, config, omega, force=.05):
        self.p = config["physical"]
        self.n = config["elements"]
        self.h = config["harmonics"]
        self.omega = float(omega)
        self.force = float(force)
        phase = np.arange(128) * 2 * np.pi / 128
        self.basis = np.column_stack([np.ones(128)] +
                      [f(j * phase) for j in range(1, self.h + 1)
                       for f in (np.cos, np.sin)])
        self.projector = self.basis.T / 128
        self.projector[1:] *= 2
        self.responses = [np.ones(self.n + 1, complex)]
        self.loads = [complex(self.p["mu"])]
        n = self.n
        mass = mass_weights(n)
        for j in range(1, self.h + 1):
            s = j * self.omega
            d = 1 + 1j * self.p["eta"] * s
            rr = -s * s + 1j * self.p["gamma"] * s
            diag = 2 * n * d * np.ones(n + 1) + rr * mass
            diag[[0, -1]] -= n * d
            ab = np.zeros((3, n + 1), complex)
            ab[1] = diag
            ab[0, 1:] = -n * d
            ab[2, :-1] = -n * d
            force_vector = np.zeros(n + 1, complex)
            force_vector[0] = rr
            response = solve_banded((1, 1), ab, force_vector)
            self.responses.append(response)
            self.loads.append(self.p["mu"] * d * response[0])

    def residual(self, coefficients):
        p = self.p
        out = p["alpha_scaled"] * (
            self.projector @ (self.basis @ coefficients) ** 3)
        out[0] += (p["k"] + self.loads[0].real) * coefficients[0]
        for j in range(1, self.h + 1):
            s = j * self.omega
            z = (p["k"] - s * s + 1j * p["c"] * s + self.loads[j])
            z *= coefficients[2*j-1] - 1j * coefficients[2*j]
            out[2*j-1] += z.real
            out[2*j] -= z.imag
        out[1] -= self.force / p["xs"]
        return out / p["k"]

    def solve(self, amplitude_choice):
        """Select smallest/largest converged amplitude at a fixed frequency.

        These labels select initial/snapshot amplitudes, not a new stability
        classification. Historical selected outer responses guide the choice.
        """
        linear = (self.p["k"] - self.omega**2 +
                  1j * self.p["c"] * self.omega + self.loads[1])
        a = .75 * self.p["alpha_scaled"]
        drive = self.force / self.p["xs"]
        squared = np.roots([a*a, 2*a*linear.real, abs(linear)**2, -drive**2])
        guesses = [np.zeros(2 * self.h + 1)]
        for value in squared:
            if abs(value.imag) < 1e-7 and value.real > 0:
                z = drive / (linear + a * value.real)
                guess = guesses[0].copy()
                guess[1], guess[2] = z.real, -z.imag
                guesses.append(guess)
        # Additional deterministic seeds handle shifts due to higher harmonics.
        for amplitude in (.1, .3, .7, 1., 1.5, 2., 3., 4.):
            z = drive / (linear + a * amplitude**2)
            guess = guesses[0].copy()
            guess[1], guess[2] = z.real, -z.imag
            guesses.append(guess)
        solutions = []
        for guess in guesses:
            result = root(self.residual, guess, tol=1e-11)
            if np.isfinite(result.x).all() and np.max(abs(self.residual(result.x))) < 1e-8:
                if not any(np.linalg.norm(result.x - old) < 1e-5 for old in solutions):
                    solutions.append(result.x)
        if not solutions:
            raise RuntimeError(f"No periodic snapshot solution at omega={self.omega}")
        solutions.sort(key=lambda x: np.ptp(self.basis @ x))
        chosen = solutions[-1 if amplitude_choice == "high" else 0]
        report = dict(omega=self.omega, amplitude_choice=amplitude_choice,
                      converged_roots=len(solutions),
                      residual=float(np.max(abs(self.residual(chosen)))),
                      amplitude=float(np.ptp(self.basis @ chosen)/2))
        return chosen, report

    def states(self, coefficients, points):
        phase = np.arange(points) * 2 * np.pi / points
        b = np.column_stack([np.ones(points)] +
             [f(j * phase) for j in range(1, self.h + 1) for f in (np.cos, np.sin)])
        derivative = np.zeros((2*self.h+1, 2*self.h+1))
        for j in range(1, self.h + 1):
            derivative[2*j-1, 2*j] = j
            derivative[2*j, 2*j-1] = -j
        pressure = np.zeros((self.n + 1, 2*self.h+1))
        pressure[:, 0] = self.p["xs"] * coefficients[0]
        for j in range(1, self.h + 1):
            z = self.p["xs"] * self.responses[j] * (
                coefficients[2*j-1] - 1j*coefficients[2*j])
            pressure[:, 2*j-1], pressure[:, 2*j] = z.real, -z.imag
        return np.column_stack([b @ coefficients, self.omega * b @ derivative @ coefficients,
                                 b @ pressure.T, self.omega * b @ derivative @ pressure.T])

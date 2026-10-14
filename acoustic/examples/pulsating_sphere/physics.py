"""Coupled breathing sphere and quadratic radial FEM; no import-time writes."""
from math import comb
import numpy as np
from scipy.linalg import eig, solve
from scipy.sparse import lil_matrix


def constants(settings):
    p = dict(settings)
    p['omega0'] = 2 * np.pi * p['frequency_hz']
    p['stiffness'] = p['mass_kg'] * p['omega0'] ** 2
    p['damping'] = 2 * p['damping_ratio'] * p['mass_kg'] * p['omega0']
    p['wavelength'] = p['sound_speed_m_s'] / p['frequency_hz']
    p['displacement_scale'] = p['force_reference_N'] / p['stiffness']
    p['velocity_scale'] = p['omega0'] * p['displacement_scale']
    p['pressure_scale'] = p['density_kg_m3'] * p['sound_speed_m_s'] * p['velocity_scale']
    p['gamma'] = (4 * np.pi * p['radius_m'] ** 2 * p['density_kg_m3']
                  * p['sound_speed_m_s'] / (p['mass_kg'] * p['omega0']))
    p['beta'] = p['sound_speed_m_s'] / (p['radius_m'] * p['omega0'])
    return p


def force(time_s, spec, p):
    """Instantaneous force / Fref, with a causal smooth exponential onset."""
    time_s = np.asarray(time_s)
    tau = np.maximum(time_s, 0) * p['omega0']
    envelope = (-np.expm1(-tau * spec['ratio'] / (2 * np.pi * spec['ramp_cycles']))) ** 4
    return (spec['amplitude'] * envelope * np.sin(spec['ratio'] * tau + spec['phase'])
            * (time_s > 0))


def assemble(elements, p):
    """Second-order FEM on z=(r-a)/lambda0 in [0,4], including the structure."""
    alpha = 1 / (2 * np.pi)
    count = 2 * elements + 1
    length = p['radial_wavelengths']
    h = length / elements
    z = np.linspace(0, length, count)
    mass = lil_matrix((count, count))
    stiffness = lil_matrix((count, count))
    me = h / 30 * np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]])
    ke = 1 / (3 * h) * np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]])
    for element in range(elements):
        ids = np.arange(2 * element, 2 * element + 3)
        for i, row in enumerate(ids):
            for j, col in enumerate(ids):
                mass[row, col] += me[i, j]
                stiffness[row, col] += ke[i, j]
    weight = 2 * np.pi * p['gamma'] / alpha ** 2
    M = np.zeros((count + 1, count + 1))
    K = np.zeros_like(M)
    D = np.zeros_like(M)
    b = np.zeros(count + 1)
    b[0] = 1
    M[0, 0] = K[0, 0] = 1
    M[1:, 1:] = weight * mass.toarray()
    K[1:, 1:] = weight * alpha ** 2 * stiffness.toarray()
    K[1, 1] += weight * alpha ** 2 * p['wavelength'] / p['radius_m']
    D[0, 0] = 2 * p['damping_ratio']
    D[0, 1] = -2 * np.pi * p['gamma']
    D[1, 0] = 2 * np.pi * p['gamma']
    D[-1, -1] = weight * alpha
    return dict(M=M, K=K, D=D, b=b, z=z)


def realization(fem):
    size = len(fem['b'])
    inverse_products = solve(fem['M'], np.c_[fem['K'], fem['D'], fem['b']], assume_a='pos')
    system = np.block([[np.zeros((size, size)), np.eye(size)],
                       [-inverse_products[:, :size], -inverse_products[:, size:2 * size]]])
    drive = np.r_[np.zeros(size), inverse_products[:, -1]]
    eigenvalues, vectors = eig(system)
    return system, drive, eigenvalues, vectors, np.linalg.inv(vectors)


def history(real, time_s, spec, p):
    """Exact solution of the semidiscrete FEM for this exponential-sum forcing.

    This solves the discretized FEM system, not the continuous spherical solution.
    All structural and acoustic initial coordinates and velocities are zero.
    """
    system, drive, eigenvalues, vectors, inverse = real
    tau = time_s * p['omega0']
    output = np.zeros((len(time_s), len(drive)), dtype=complex)
    exponential = np.exp(tau[:, None] * eigenvalues)
    for j in range(5):
        lam = -j * spec['ratio'] / (2 * np.pi * spec['ramp_cycles']) + 1j * spec['ratio']
        particular = np.linalg.solve(lam * np.eye(len(drive)) - system, drive)
        output += (-1) ** j * comb(4, j) * (
            np.exp(lam * tau)[:, None] * particular
            - (exponential * (inverse @ particular)[None, :]) @ vectors.T)
    return spec['amplitude'] * np.imag(np.exp(1j * spec['phase']) * output)


def exact_state(time_s, spec, p):
    """Continuous analytical three-state solution, used only after training."""
    L = np.array([[0, 1, 0], [-1, -2 * p['damping_ratio'], -p['gamma']],
                  [-1, -2 * p['damping_ratio'], -p['gamma'] - p['beta']]])
    drive = np.array([0., 1., 1.])
    eigenvalues, vectors = np.linalg.eig(L)
    inverse = np.linalg.inv(vectors)
    times = np.asarray(time_s)
    tau = np.maximum(times.ravel(), 0) * p['omega0']
    output = np.zeros((len(tau), 3), dtype=complex)
    for j in range(5):
        lam = -j * spec['ratio'] / (2 * np.pi * spec['ramp_cycles']) + 1j * spec['ratio']
        particular = np.linalg.solve(lam * np.eye(3) - L, drive)
        transient = (np.exp(tau[:, None] * eigenvalues) * (inverse @ particular)[None, :]) @ vectors.T
        output += (-1) ** j * comb(4, j) * (np.exp(lam * tau)[:, None] * particular - transient)
    return (spec['amplitude'] * np.imag(np.exp(1j * spec['phase']) * output)
            * (times.ravel() > 0)[:, None]).reshape(times.shape + (3,))

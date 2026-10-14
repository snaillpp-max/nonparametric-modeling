"""POD projection, FEM-recomputed labels, and the original fixed-hidden map."""
import numpy as np
from scipy.linalg import lstsq
from numba import njit
from physics import fem_rhs


def project(states, phi, weights, xs):
    n = len(weights) - 1
    wp = weights[:, None] * phi
    return np.column_stack([states[:, 0],
        (states[:, 2:n+3] - xs*states[:, 0, None]) @ wp, states[:, 1],
        (states[:, n+3:] - xs*states[:, 1, None]) @ wp])


def labels(inputs, phi, weights, par, chunk=256):
    """Lift -> recompute full FEM acceleration -> project; never add old labels."""
    n = len(weights) - 1
    xs, c, k, alpha, mu, eta, gamma = par[4], par[2], par[1], par[3], par[6], par[7], par[8]
    wp = weights[:, None] * phi
    rows = []
    for first in range(0, len(inputs), chunk):
        z = inputs[first:first+chunk]
        x, v = z[:, 0], z[:, 33]
        p = z[:, 1:33] @ phi.T + xs * x[:, None]
        pt = z[:, 34:66] @ phi.T + xs * v[:, None]
        h = p + eta * pt
        acc = z[:, 66]/xs - c*v - k*x - alpha*x**3 - mu/xs*h[:, 0]
        ptt = np.empty_like(p)
        ptt[:, 1:-1] = n*n*(h[:, :-2] - 2*h[:, 1:-1] + h[:, 2:])
        ptt[:, 0] = 2*n*n*(h[:, 1] - h[:, 0]) + 2*n*xs*(acc + gamma*v)
        ptt[:, -1] = 2*n*n*(h[:, -2] - h[:, -1])
        ptt -= gamma * pt
        rows.append(np.column_stack([acc, (ptt-xs*acc[:, None]) @ wp]))
    return np.vstack(rows)


def label_check(inputs, phi, weights, par):
    truth = labels(inputs[:5], phi, weights, par)
    differences = []
    for inp, expected in zip(inputs[:5], truth):
        xs = par[4]
        full = np.r_[inp[0], inp[33], xs*inp[0]+phi@inp[1:33],
                     xs*inp[33]+phi@inp[34:66]]
        result = fem_rhs(full, inp[-1], par)
        reference = np.r_[result[1], (result[len(weights)+2:]-xs*result[1]) @
                          (weights[:, None]*phi)]
        differences.append(float(np.linalg.norm(reference-expected) /
                                 max(np.linalg.norm(reference), 1e-30)))
    return max(differences)


def design(normalized, w0, w1):
    hidden = np.tanh(np.tanh(normalized[:, 0, None]*w0[None, :]) @ w1.T)
    return np.column_stack([normalized, hidden, np.ones(len(normalized))])


def fit(train_x, train_y, val_x, val_y, config):
    xs = np.maximum(np.sqrt(np.mean(train_x*train_x, axis=0)), 1e-10)
    ys = np.maximum(np.sqrt(np.mean(train_y*train_y, axis=0)), 1e-10)
    x, y = train_x/xs, train_y/ys
    vx, vy = val_x/xs, val_y/ys
    rng = np.random.default_rng(config["seed_hidden"])
    directions = rng.uniform(.5, 1.5, 64)*rng.choice([-1, 1], 64)
    mixing = rng.normal(size=(32, 64))
    mixing /= np.linalg.norm(mixing, axis=1)[:, None]
    w1 = .5*mixing
    best, candidates = None, []
    for scale in config["feature_scales"]:
        w0 = scale*directions
        coef, _, rank, _ = lstsq(design(x, w0, w1), y,
                                 cond=config["svd_cutoff"], lapack_driver="gelsd")
        prediction = design(vx, w0, w1) @ coef
        error = np.linalg.norm(prediction-vy, axis=0) / np.maximum(
            np.linalg.norm(vy, axis=0), 1e-30)
        score = float(max(error))
        candidates.append(dict(scale=scale, numerical_rank=int(rank),
                               validation_max_relative_l2=score))
        if best is None or score < best[0]:
            best = score, w0, coef, scale
    _, w0, coef, selected = best
    return dict(xs=xs, ys=ys, w0=w0, w1=w1, coef=coef), dict(
        architecture=[67, 64, 32, 33], fitted_parameters=3300,
        fixed_hidden=True, nonlinear_input="X only",
        affine_skip="all 67 normalized state and force inputs plus bias",
        selected_scale=selected, candidates=candidates)


@njit(cache=True)
def acceleration(state, force, xs, ys, w0, w1, coef):
    x = np.empty(67)
    x[:66] = state
    x[66] = force
    x /= xs
    hidden = np.tanh(w1 @ np.tanh(w0*x[0]))
    return (x @ coef[:67] + hidden @ coef[67:99] + coef[99])*ys


@njit(cache=True)
def nn_rhs(state, time, omega, force, xs, ys, w0, w1, coef):
    out = np.empty(66)
    out[:33] = state[33:]
    out[33:] = acceleration(state, force*np.cos(omega*time), xs, ys, w0, w1, coef)
    return out

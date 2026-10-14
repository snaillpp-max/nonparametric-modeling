"""Stage 3: fixed Tanh features, ridge readout and offline Schur correction."""
import json
import numpy as np
from scipy.linalg import lstsq, schur, solve_continuous_are
from scipy.integrate import solve_ivp
from physics import force


def features(epsilon, rank, seed):
    """Same 128/64 block features as the original POD24 implementation."""
    rng = np.random.default_rng(seed)
    w1 = np.zeros((128, 2 * rank + 1))
    for rows, cols in [(slice(0, 56), slice(0, rank)),
                       (slice(56, 112), slice(rank, 2 * rank))]:
        direction = rng.normal(size=(56, rank))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        w1[rows, cols] = epsilon * direction
    w1[112:, 2 * rank] = epsilon * rng.uniform(.4, 1., 16) * rng.choice([-1., 1.], 16)
    w2 = np.zeros((64, 128))
    for rows, cols, shape in [(slice(0, 28), slice(0, 56), (28, 56)),
                              (slice(28, 56), slice(56, 112), (28, 56)),
                              (slice(56, 64), slice(112, 128), (8, 16))]:
        mixing = rng.normal(size=shape)
        mixing *= .5 / np.linalg.norm(mixing, axis=1)[:, None]
        w2[rows, cols] = mixing
    return w1, w2


def acceleration(values, model):
    h = np.asarray(values) / model['input_scale']
    h = np.tanh(h @ model['W1'].T)
    h = np.tanh(h @ model['W2'].T)
    return (h @ model['W3'].T) * model['output_scale']


def rollout(times, spec, model, p, integrator, tight=False):
    rank = len(model['output_scale'])
    def rhs(tau, state):
        return np.r_[state[rank:], acceleration(np.r_[state, force(tau / p['omega0'], spec, p)], model)]
    def stop(tau, state):
        return 1e6 - np.linalg.norm(state / model['input_scale'][:2 * rank])
    stop.terminal = True
    stop.direction = -1
    solution = solve_ivp(rhs, (0, times[-1] * p['omega0']), np.zeros(2 * rank),
                         t_eval=times * p['omega0'], events=stop, method='DOP853',
                         rtol=integrator['rtol'] / (100 if tight else 1),
                         atol=integrator['atol'] / (100 if tight else 1),
                         max_step=integrator['max_step_tau'] / (2 if tight else 1))
    if not solution.success or len(solution.t) != len(times) or not np.isfinite(solution.y).all():
        return None
    return solution.y.T


def companion(jacobian, rank):
    return np.block([[np.zeros((rank, rank)), np.eye(rank)],
                     [jacobian[:, :rank], jacobian[:, rank:2 * rank]]])


def run(output, config, profile, p):
    settings = config['profiles'][profile]
    rank = settings['pod_rank']
    with np.load(output / 'training_pairs.npz') as archive:
        x, y, xv, yv = [archive[k] for k in ['x', 'y', 'validation_x', 'validation_y']]
    with np.load(output / 'pod.npz') as archive:
        basis = archive['basis']
    input_scale = np.sqrt(np.mean(x * x, axis=0))
    output_scale = np.sqrt(np.mean(y * y, axis=0))
    if np.any(input_scale <= 0) or np.any(output_scale <= 0):
        raise ValueError('Nonpositive training RMS scale')
    metadata = json.loads((output / 'generation.json').read_text(encoding='utf-8'))
    validation = []
    for spec in metadata['validation']:
        with np.load(output / spec['file']) as archive:
            validation.append((spec, archive['time_s'], archive['states']))
    best = None
    records = []
    for epsilon in settings['feature_epsilons']:
        w1, w2 = features(epsilon, rank, config['feature_seed'])
        hidden = np.tanh(np.tanh(x / input_scale @ w1.T) @ w2.T)
        left, singular, right = np.linalg.svd(hidden, full_matrices=False)
        tangent = w2 @ w1 / input_scale
        tangent_inverse = np.linalg.pinv(tangent, rcond=1e-13)
        for ridge in settings['ridge_ratios']:
            if ridge == 0:
                coefficient = lstsq(hidden, y / output_scale, cond=1e-12)[0]
            else:
                coefficient = (right.T * (singular / (singular ** 2 + ridge * singular[0] ** 2))) @ (left.T @ (y / output_scale))
            jacobian = (coefficient.T @ tangent) * output_scale[:, None]
            system = companion(jacobian, rank)
            original_growth = float(np.linalg.eigvals(system).real.max())
            readout = coefficient.T
            corrected_dimensions = 0
            # Offline correction of the learned zero-state Jacobian. It changes
            # only the output weights; no FEM/Galerkin operator fills NN weights.
            if original_growth > 1e-7:
                state_scale = input_scale[:2 * rank]
                normalized = system * state_scale[None, :] / state_scale[:, None]
                triangular, rotation, stable_dimension = schur(
                    normalized, output='real', sort=lambda re, im: re < -1e-7)
                corrected_dimensions = 2 * rank - stable_dimension
                actuation = rotation[rank:, stable_dimension:].T
                try:
                    riccati = solve_continuous_are(
                        triangular[stable_dimension:, stable_dimension:], actuation,
                        1e-10 * np.eye(corrected_dimensions), np.eye(rank))
                except np.linalg.LinAlgError:
                    records.append(dict(epsilon=epsilon, ridge=ridge, rejected='Riccati solve failed'))
                    continue
                target = jacobian.copy()
                target[:, :2 * rank] -= ((actuation.T @ riccati @ rotation[:, stable_dimension:].T)
                                         * state_scale[rank:, None] / state_scale[None, :])
                readout = coefficient.T + (target / output_scale[:, None] - coefficient.T @ tangent) @ tangent_inverse
            model = dict(W1=w1, W2=w2, W3=readout, input_scale=input_scale, output_scale=output_scale)
            achieved = (readout @ tangent) * output_scale[:, None]
            corrected_growth = float(np.linalg.eigvals(companion(achieved, rank)).real.max())
            errors = []
            for spec, times, reference in validation:
                prediction = rollout(times, spec, model, p, config['integrator'])
                if prediction is None:
                    errors.append(float('inf'))
                    continue
                predicted_velocity = prediction[:, rank:] @ basis.T
                size = basis.shape[0]
                reference_velocity = reference[:, size:]
                errors.append(float(np.linalg.norm(predicted_velocity[:, 1:] - reference_velocity[:, 1:])
                                    / np.linalg.norm(reference_velocity[:, 1:])))
            item = dict(epsilon=epsilon, ridge=ridge, original_growth=original_growth,
                        corrected_growth=corrected_growth, corrected_schur_dimensions=corrected_dimensions,
                        validation_field_errors=errors,
                        validation_acceleration_error=float(np.linalg.norm(acceleration(xv, model) - yv) / np.linalg.norm(yv)))
            records.append(item)
            score = max(errors)
            print(f'train: epsilon={epsilon:g}, ridge={ridge:g}, validation={score:.6g}', flush=True)
            if np.isfinite(score) and (best is None or score < best[0]):
                best = (score, model, item)
    (output / 'selection.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    if best is None:
        raise RuntimeError('No candidate completed validation integration')
    if profile == 'paper' and best[0] >= .02:
        raise RuntimeError('Paper-profile validation error exceeds the original 2% guard')
    if not all(np.isfinite(a).all() for a in best[1].values()):
        raise RuntimeError('Nonfinite selected network parameters')
    best[2].update(source_sha256=config['_source_hashes'], feature_seed=config['feature_seed'],
                   architecture=[2 * rank + 1, 128, 64, rank], weights_finite=True,
                   training_x_shape=list(x.shape), training_y_shape=list(y.shape),
                   validation_x_shape=list(xv.shape), validation_y_shape=list(yv.shape),
                   training_method='fixed Tanh features + ridge readout + offline learned-Jacobian Schur correction')
    np.savez_compressed(output / 'model.npz', basis=basis, **best[1])
    (output / 'selected.json').write_text(json.dumps(best[2], indent=2), encoding='utf-8')

"""Stage 2: mass POD, physical amplitude augmentation and derivative labels."""
import json
import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular
from physics import force


def derivative_pairs(time_s, coordinates, spec, p):
    dt = (time_s[1] - time_s[0]) * p['omega0']
    velocity = (coordinates[:-4] - 8 * coordinates[1:-3] + 8 * coordinates[3:-1]
                - coordinates[4:]) / (12 * dt)
    acceleration = (-coordinates[4:] + 16 * coordinates[3:-1] - 30 * coordinates[2:-2]
                    + 16 * coordinates[1:-3] - coordinates[:-4]) / (12 * dt ** 2)
    return np.c_[coordinates[2:-2], velocity, force(time_s, spec, p)[2:-2]], acceleration


def run(output, config, profile, p):
    settings = config['profiles'][profile]
    metadata = json.loads((output / 'generation.json').read_text(encoding='utf-8'))
    with np.load(output / 'fem.npz') as archive:
        mass = archive['M']
    size = len(mass)
    factor = cholesky(mass, lower=True)
    covariance = np.zeros_like(mass)
    stride = settings['sample_stride']
    for spec in metadata['training']:
        with np.load(output / spec['file']) as archive:
            snapshots = archive['states'][::stride, :size]
        weighted = snapshots @ factor
        covariance += weighted.T @ weighted / len(weighted)
    eigenvalues, vectors = eigh(covariance)
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    vectors = vectors[:, ::-1]
    rank = settings['pod_rank']
    if rank >= size:
        raise ValueError('POD rank must be smaller than the full number of coordinates')
    basis = solve_triangular(factor.T, vectors[:, :rank], lower=False)
    orthogonality = float(np.max(abs(basis.T @ mass @ basis - np.eye(rank))))
    if orthogonality > 1e-10:
        raise RuntimeError('Mass-orthonormality check failed')
    np.savez_compressed(output / 'pod.npz', basis=basis, eigenvalues=eigenvalues)
    pairs = []
    for spec in metadata['training']:
        with np.load(output / spec['file']) as archive:
            times = archive['time_s']
            coordinates = archive['states'][:, :size] @ mass @ basis
        for amplitude in settings['training_amplitudes']:
            scaled = dict(spec, amplitude=amplitude)
            x, y = derivative_pairs(times, amplitude * coordinates, scaled, p)
            pairs.append((x[::stride], y[::stride]))
    x = np.concatenate([entry[0] for entry in pairs])
    y = np.concatenate([entry[1] for entry in pairs])
    validation = []
    for spec in metadata['validation']:
        with np.load(output / spec['file']) as archive:
            times = archive['time_s']
            coordinates = archive['states'][:, :size] @ mass @ basis
        xv, yv = derivative_pairs(times, coordinates, spec, p)
        validation.append((xv[::stride], yv[::stride]))
    xv = np.concatenate([entry[0] for entry in validation])
    yv = np.concatenate([entry[1] for entry in validation])
    if not all(np.isfinite(a).all() for a in [x, y, xv, yv]):
        raise RuntimeError('Nonfinite projected training/validation pairs')
    np.savez_compressed(output / 'training_pairs.npz', x=x, y=y, validation_x=xv, validation_y=yv)
    info = dict(rank=rank, retained_mass_energy=float(eigenvalues[:rank].sum() / eigenvalues.sum()),
                max_mass_orthogonality_error=orthogonality, training_samples=len(x),
                validation_samples=len(xv), input_dimension=x.shape[1], output_dimension=y.shape[1],
                training_x_shape=list(x.shape), training_y_shape=list(y.shape),
                validation_x_shape=list(xv.shape), validation_y_shape=list(yv.shape),
                pairs_finite=True, source_sha256=config['_source_hashes'],
                augmentation='linear FEM trajectory scaling at amplitudes 0.6 and 1.2; no random jitter',
                derivative_labels='five-point fourth-order finite differences in tau=omega0*t',
                centering=False, basis_source='training unit histories only')
    (output / 'pod.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'pod: rank {rank}, training {len(x)}, validation {len(xv)}', flush=True)

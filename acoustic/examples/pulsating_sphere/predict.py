"""Stage 4: one independent near-resonance NN trajectory; no frequency sweep."""
import json
import numpy as np
from physics import exact_state
from train import rollout


def run(output, config, profile, p):
    settings = config['profiles'][profile]
    with np.load(output / 'model.npz') as archive:
        model = {key: archive[key] for key in archive.files}
    with np.load(output / 'fem.npz') as archive:
        z = archive['z']
    count = settings['prediction_reference_cycles'] * settings['samples_per_reference_cycle']
    times = np.arange(count + 1) / (settings['samples_per_reference_cycle'] * p['frequency_hz'])
    spec = dict(config['prediction'])
    state = rollout(times, spec, model, p, config['integrator'])
    if state is None:
        raise RuntimeError('Independent prediction did not complete')
    rank = settings['pod_rank']
    radius = p['radius_m'] + p['wavelength'] * z
    velocity = state[:, rank:] @ model['basis'][1:].T
    pressure = -2 * np.pi * p['radius_m'] / radius * velocity
    reference = exact_state(times[:, None] - z[None, :] / p['frequency_hz'], spec, p)[:, :, 2]
    reference *= p['radius_m'] / radius
    probe = int(np.argmin(abs(z - 1)))
    error = pressure - reference
    weight = radius ** 2
    weight[[0, -1]] *= .5
    info = dict(profile=profile, prediction=spec, integration_complete=True,
                state_finite=bool(np.isfinite(state).all()), pressure_finite=bool(np.isfinite(pressure).all()),
                reference_finite=bool(np.isfinite(reference).all()), pressure_shape=list(pressure.shape),
                source_sha256=config['_source_hashes'],
                samples=len(times), independent_state_dimension=2 * rank,
                probe_radius_m=float(radius[probe]),
                probe_relative_l2=float(np.linalg.norm(error[:, probe]) / np.linalg.norm(reference[:, probe])),
                nodal_radial_weighted_relative_l2=float(np.sqrt(np.sum(error ** 2 * weight) / np.sum(reference ** 2 * weight))),
                pressure_scale_Pa=p['pressure_scale'], pressure_units='p/Pref',
                accuracy_scope=('pipeline smoke check only; not a paper-accuracy reproduction' if profile == 'smoke'
                                else 'single independent trajectory; not the published 50-frequency sweep'))
    np.savez_compressed(output / 'prediction.npz', time_s=times, radius_m=radius, reduced_state=state,
                        predicted_pressure_normalized=pressure, exact_pressure_normalized=reference,
                        predicted_pressure_Pa=pressure * p['pressure_scale'], exact_pressure_Pa=reference * p['pressure_scale'])
    (output / 'prediction_metrics.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=2), flush=True)

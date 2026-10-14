"""Stage 1: generate independent training and validation FEM trajectories."""
import json
import numpy as np
from physics import assemble, realization, history


def run(output, config, profile, p):
    settings = config['profiles'][profile]
    fem = assemble(settings['elements'], p)
    np.savez_compressed(output / 'fem.npz', **fem)
    real = realization(fem)
    time_s = np.arange(settings['training_reference_cycles'] * settings['samples_per_reference_cycle'] + 1)
    time_s = time_s / (settings['samples_per_reference_cycle'] * p['frequency_hz'])
    training = []
    for ratio in settings['training_ratios']:
        for phase in settings['training_phases']:
            spec = dict(ratio=ratio, amplitude=1., ramp_cycles=settings['training_ramp_cycles'], phase=phase)
            filename = f'train_unit_{len(training):02d}.npz'
            states = history(real, time_s, spec, p)
            if not np.isfinite(states).all():
                raise RuntimeError('Nonfinite generated FEM training history')
            np.savez_compressed(output / filename, time_s=time_s, states=states)
            training.append(dict(file=filename, **spec))
    validation = []
    for ratio in settings['validation_ratios']:
        spec = dict(ratio=ratio, amplitude=.9, ramp_cycles=.35, phase=.4)
        filename = f'validation_{len(validation):02d}.npz'
        states = history(real, time_s, spec, p)
        if not np.isfinite(states).all():
            raise RuntimeError('Nonfinite generated FEM validation history')
        np.savez_compressed(output / filename, time_s=time_s, states=states)
        validation.append(dict(file=filename, **spec))
    metadata = dict(profile=profile, physics=config['physics'], settings=settings,
                    training=training, validation=validation,
                    pipeline_config=config['_pipeline_identity'], source_sha256=config['_source_hashes'],
                    config_sha256=config['_config_sha256'], sample_count=len(time_s),
                    full_coordinate_count=len(fem['b']), acoustic_node_count=len(fem['z']),
                    full_state_shape=[len(time_s), 2 * len(fem['b'])], generated_states_finite=True,
                    source='quadratic FEM; zero initial state; exact semidiscrete time response')
    (output / 'generation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'generate: {len(training)} unit histories and {len(validation)} validation histories', flush=True)

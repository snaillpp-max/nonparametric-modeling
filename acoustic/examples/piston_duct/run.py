"""Portable four-stage piston-duct workflow; default profile is paper."""
import argparse
import hashlib
import platform
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="paper")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"output")
    parser.add_argument("--stage", choices=("all", "generate", "pod", "train", "predict"), default="all")
    parser.add_argument("--cycles", type=int, help="override prediction cycles, e.g. 3000")
    return parser.parse_args()


ARGS = parse_args()
OUT = ARGS.output.expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
RUNTIME = OUT/"runtime"
for name in ("tmp", "numba_cache", "mplconfig"):
    (RUNTIME/name).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(RUNTIME/"tmp")
os.environ["TMP"] = str(RUNTIME/"tmp")
os.environ["NUMBA_CACHE_DIR"] = str(RUNTIME/"numba_cache")
os.environ["MPLCONFIGDIR"] = str(RUNTIME/"mplconfig")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import scipy
import numba
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from physics import parameters, mass_weights, sample_fem, PeriodicSnapshots
from network import project, labels, label_check, fit, design, nn_rhs


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def log(stage, **details):
    print(json.dumps(dict(stage=stage, **details)), flush=True)


def configuration():
    raw = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
    config = dict(raw["common"], **raw["profiles"][ARGS.profile], physical=raw["physical"],
                  profile=ARGS.profile)
    if ARGS.cycles is not None:
        if ARGS.cycles < 1:
            raise ValueError("--cycles must be positive")
        config["prediction_cycles"] = ARGS.cycles
    if config["physical"]["beta"] != 0:
        raise ValueError("This example requires the documented linear duct: beta=0")
    for key in ("fem_steps_per_cycle", "base_fem_steps_per_cycle"):
        if config[key] % config["recorded_points_per_cycle"]:
            raise ValueError("FEM steps must be divisible by recorded points")
    saved = OUT/"config_used.json"
    if saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Output contains another configuration; choose a fresh --output")
    source_hashes = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("config.json", "physics.py", "network.py", "run.py")}
    provenance = dict(source_sha256=source_hashes,
        effective_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        profile=ARGS.profile, python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__)
    existing = OUT/"provenance.json"
    if existing.exists():
        old = json.loads(existing.read_text(encoding="utf-8"))
        if old["source_sha256"] != source_hashes or old["effective_config_sha256"] != provenance["effective_config_sha256"]:
            raise ValueError("Output source/config hashes conflict; choose a fresh --output")
    save_json(saved, config)
    save_json(existing, provenance)
    return config


def generate(config):
    directory = OUT/"data"
    directory.mkdir(exist_ok=True)
    par = parameters(config)
    n, per = config["elements"], config["recorded_points_per_cycle"]
    cases, checks = [], []
    cache = {}

    def periodic(omega, choice):
        key = (float(omega), choice)
        if key not in cache:
            solver = PeriodicSnapshots(config, omega)
            coefficients, report = solver.solve(choice)
            cache[key] = solver.states(coefficients, per)
            checks.append(report)
        return cache[key]

    def store_case(name, values, omega, kind, force=.05):
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite FEM data: {name}")
        filename = name+".npy"
        np.save(directory/filename, values.astype(np.float32))
        cases.append(dict(name=name, filename=filename, omega=float(omega),
                          kind=kind, force=float(force), samples=len(values)))
        log("generate_case", name=name, rows=len(values))

    for omega in np.linspace(43, 52, 19):
        choices = ("high", "low") if 46.02 < omega < 49.74 else ("low",)
        for choice in choices:
            store_case(f"steady_{choice}_{omega:.3f}", periodic(omega, choice),
                       omega, "steady")
    high = periodic(48, "high")[0].copy()
    low = periodic(48, "low")[0].copy()
    # Historical finite-frequency loading changes; no fold finding is performed.
    jobs = [
        ("startup", 48., np.zeros(2*n+4), "transient", .05),
        ("jump_up", 49.79358794812716, periodic(49.69358794812716, "high")[0],
         "transient", .05),
        ("jump_down", 45.960437141481314, periodic(46.060437141481314, "low")[0],
         "transient", .05),
        ("high_48", 48., high, "steady_trajectory", .05),
        ("low_48", 48., low, "steady_trajectory", .05)]
    for force in (.04, .045, .055, .06):
        for choice, initial in (("high", high), ("low", low)):
            jobs.append((f"F{force:.3f}_{choice}_w48", 48., initial,
                         "amplitude_transition", force))
    for name, omega, initial, kind, force in jobs:
        spp = (config["base_fem_steps_per_cycle"] if force == .05 else
               config["fem_steps_per_cycle"])
        states, final = sample_fem(np.asarray(initial, float), omega, force, par,
                config["trajectory_cycles"], spp, per)
        if not np.isfinite(final).all():
            raise FloatingPointError(f"Nonfinite terminal FEM state: {name}")
        store_case(name, states, omega, kind, force)
        cases[-1]["fem_steps_per_cycle"] = spp
        cases[-1]["cycles"] = config["trajectory_cycles"]
    weights = mass_weights(n)
    mean_error = max(float(np.max(abs(np.load(directory/c["filename"])[:, 2:n+3] @
                          weights - config["physical"]["xs"] *
                          np.load(directory/c["filename"])[:, 0]))) for c in cases)
    manifest = dict(cases=cases, case_count=len(cases),
                    sample_count=sum(c["samples"] for c in cases), finite=True,
                    periodic_snapshot_checks=checks,
                    max_mean_pressure_constraint_absolute_error=mean_error,
                    historical_difference="Fixed-frequency independent harmonic solves; initial states are synthesized on the selected mesh, not imported historical settled 1024-element seeds. Regenerated weights are not claimed bitwise identical.",
                    smoke=ARGS.profile=="smoke")
    save_json(directory/"manifest.json", manifest)


def pod(config):
    directory = OUT/"pod"
    directory.mkdir(exist_ok=True)
    cases = json.loads((OUT/"data"/"manifest.json").read_text())["cases"]
    n, rank = config["elements"], config["pod_rank"]
    xs = config["physical"]["xs"]
    weights = mass_weights(n)
    covariance = np.zeros((n+1, n+1))
    random = np.random.default_rng(config["seed_split"])
    splits = {}
    for index, case in enumerate(cases):
        states = np.load(OUT/"data"/case["filename"], mmap_mode="r")
        rng = random if index < 31 else np.random.default_rng(config["seed_data"]+index)
        indices = rng.permutation(len(states))
        first, second = int(.7*len(states)), int(.85*len(states))
        split = dict(train=indices[:first], val=indices[first:second], test=indices[second:])
        splits[case["name"]] = split
        np.savez(directory/(case["name"]+"_split.npz"), **split)
        # Historical POD uses the original 31 cases only.
        if index >= 31:
            continue
        training = np.asarray(states[split["train"]], float)
        pressure = training[:, 2:n+3]-xs*training[:, 0, None]
        velocity = (training[:, n+3:]-xs*training[:, 1, None])/48
        pressure -= (pressure @ weights)[:, None]
        velocity -= (velocity @ weights)[:, None]
        energy = np.sum(pressure**2*weights)+np.sum(velocity**2*weights)
        pressure *= np.sqrt(weights/max(energy, 1e-30))
        velocity *= np.sqrt(weights/max(energy, 1e-30))
        covariance += pressure.T @ pressure + velocity.T @ velocity
    vals, vectors = eigh(covariance, subset_by_index=(n+1-rank, n))
    vals, phi = np.maximum(vals[::-1], 0), vectors[:, ::-1]/np.sqrt(weights[:, None])
    phi -= np.ones((n+1, 1)) @ (weights @ phi)[None, :]
    q, _ = np.linalg.qr(np.sqrt(weights[:, None])*phi)
    phi = q/np.sqrt(weights[:, None])
    orth = float(np.linalg.norm(phi.T@(weights[:, None]*phi)-np.eye(rank)))
    np.savez(directory/"basis.npz", phi=phi, weights=weights, eigenvalues=vals,
             retained_energy=vals.sum()/np.trace(covariance))
    par = parameters(config)
    packs = {s:[] for s in ("train", "val", "test")}
    targets = {s:[] for s in packs}
    augmented_x, augmented_y, details = [], [], []
    for index, case in enumerate(cases):
        states = np.asarray(np.load(OUT/"data"/case["filename"]), float)
        reduced = project(states, phi, weights, xs)
        force = case["force"]*np.cos(np.arange(len(states))*2*np.pi /
                                    config["recorded_points_per_cycle"])
        inputs = np.column_stack([reduced, force])
        output = labels(inputs, phi, weights, par)
        split = splits[case["name"]]
        for name in packs:
            packs[name].append(inputs[split[name]])
            targets[name].append(output[split[name]])
        rng = np.random.default_rng(config["seed_data"]+1000+index)
        parents = rng.choice(split["train"], min(config["augmentation_parents"],
                                                len(split["train"])), replace=False)
        parents = np.repeat(parents, config["augmentation_repeats"])
        perturbed = inputs[parents].copy()
        rms = np.sqrt(np.mean(inputs[split["train"], :66]**2, axis=0))
        scale = np.maximum(abs(perturbed[:, :66]), config["rms_floor_fraction"]*rms)
        perturbed[:, :66] += rng.uniform(-config["augmentation_fraction"],
                        config["augmentation_fraction"], scale.shape)*scale
        augmented_x.append(perturbed)
        augmented_y.append(labels(perturbed, phi, weights, par))
        assert not np.intersect1d(parents, np.r_[split["val"], split["test"]]).size
        np.savez_compressed(directory/(case["name"]+"_augmentation.npz"),
                           input=perturbed, target=augmented_y[-1], parent_indices=parents)
        details.append(dict(name=case["name"], counts={s:len(split[s]) for s in split},
                            augmented=len(parents), force_preserved=True))
    counts = {}
    for name in packs:
        x = np.vstack(packs[name]+(augmented_x if name=="train" else []))
        y = np.vstack(targets[name]+(augmented_y if name=="train" else []))
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise FloatingPointError("Nonfinite reduced training data")
        np.save(directory/(name+"_input.npy"), x)
        np.save(directory/(name+"_target.npy"), y)
        counts[name] = len(x)
    check = label_check(np.load(directory/"test_input.npy"), phi, weights, par)
    if check > 1e-10 or orth > 1e-10:
        raise AssertionError("POD orthogonality / FEM label check failed")
    save_json(directory/"manifest.json", dict(rank=rank, state_dimension=66,
        input_dimension=67, output_dimension=33, counts=counts, cases=details,
        augmented=sum(len(v) for v in augmented_x), mass_orthogonality_error=orth,
        retained_energy=float(vals.sum()/np.trace(covariance)), fem_label_check=check,
        label_policy="All labels recomputed from lifted POD states through full FEM RHS",
        basis_cases=31, finite=True, sample_parameter_gate=counts["train"]>=.7*3300))
    log("pod_complete", counts=counts, label_check=check)


def train(config):
    directory = OUT/"training"
    directory.mkdir(exist_ok=True)
    source = OUT/"pod"
    weights, report = fit(np.load(source/"train_input.npy"),
                          np.load(source/"train_target.npy"),
                          np.load(source/"val_input.npy"),
                          np.load(source/"val_target.npy"), config)
    basis = np.load(source/"basis.npz")
    np.savez(directory/"model.npz", **weights, phi=basis["phi"], weights=basis["weights"],
             piston_displacement_scale=config["physical"]["xs"])
    report["metrics"] = {}
    for split in ("val", "test"):
        inputs = np.load(source/(split+"_input.npy"))
        reference = np.load(source/(split+"_target.npy"))
        prediction = design(inputs/weights["xs"], weights["w0"], weights["w1"]) @ weights["coef"] * weights["ys"]
        error = np.linalg.norm(prediction-reference, axis=0)/np.maximum(
                np.linalg.norm(reference, axis=0), 1e-30)
        report["metrics"][split] = dict(max_relative_l2=float(max(error)),
                                        per_output_relative_l2=error.tolist())
    report["training_count"] = len(np.load(source/"train_input.npy", mmap_mode="r"))
    report["finite"] = all(np.isfinite(value).all() for value in weights.values())
    if not report["finite"]:
        raise FloatingPointError("Nonfinite trained coefficients")
    save_json(directory/"report.json", report)
    log("train_complete", selected_scale=report["selected_scale"],
        test_error=report["metrics"]["test"]["max_relative_l2"])


def predict(config):
    directory = OUT/"prediction"
    directory.mkdir(exist_ok=True)
    bundle = np.load(OUT/"training"/"model.npz")
    model = tuple(bundle[k] for k in ("xs", "ys", "w0", "w1", "coef"))
    phi, weights = bundle["phi"], bundle["weights"]
    par = parameters(config)
    n, per = config["elements"], config["recorded_points_per_cycle"]
    cycles = config["prediction_cycles"]
    reports = []
    for name in ("startup", "high_48"):
        full = np.asarray(np.load(OUT/"data"/(name+".npy"), mmap_mode="r")[0], float)
        state = project(full[None, :], phi, weights, config["physical"]["xs"])[0]
        history = np.lib.format.open_memmap(directory/(name+"_history.npy"), mode="w+",
                                           dtype=float, shape=(cycles*per, 5))
        err_p = den_p = err_x = den_x = err_mid = den_mid = 0.
        all_finite, function_evaluations = True, 0
        for done in range(0, cycles, config["prediction_block_cycles"]):
            block = min(config["prediction_block_cycles"], cycles-done)
            times = (done+np.arange(block*per+1)/per)*2*np.pi/48
            sol = solve_ivp(lambda t,z:nn_rhs(z,t,48.,.05,*model),
                (times[0],times[-1]), state, t_eval=times, method="DOP853",
                rtol=config["nn_rtol"], atol=config["nn_atol"])
            if not sol.success or not np.isfinite(sol.y).all():
                raise RuntimeError(f"NN integration failed for {name}: {sol.message}")
            function_evaluations += sol.nfev
            state = sol.y[:, -1]
            reduced = sol.y[:, :-1].T
            truth, full = sample_fem(full, 48., .05, par, block,
                         config["fem_steps_per_cycle"], per)
            if not np.isfinite(truth).all() or not np.isfinite(full).all():
                raise FloatingPointError(f"FEM integration failed for {name}")
            p = reduced[:, 1:33]@phi.T+config["physical"]["xs"]*reduced[:, 0, None]
            reference = np.asarray(truth[:, 2:n+3], float)
            diff = p-reference
            err_p += np.sum(diff**2*weights)
            den_p += np.sum(reference**2*weights)
            err_x += np.sum((reduced[:, 0]-truth[:, 0])**2)
            den_x += np.sum(truth[:, 0]**2)
            err_mid += np.sum(diff[:, n//2]**2)
            den_mid += np.sum(reference[:, n//2]**2)
            sl = slice(done*per, (done+block)*per)
            history[sl] = np.column_stack([times[:-1]*48/(2*np.pi),
                        truth[:, 0], reduced[:, 0], reference[:, n//2], p[:, n//2]])
            history.flush()
            if done+block == cycles:
                tail = min(5*per, len(p))
                np.savez_compressed(directory/(name+"_last_cycles.npz"),
                    cycles=history[sl][-tail:, 0], z=np.linspace(0, 1, n+1),
                    truth=reference[-tail:], prediction=p[-tail:], error=-diff[-tail:])
            log("predict_block", case=name, completed_cycles=done+block, target=cycles)
        reports.append(dict(name=name, finite=all_finite, cycles=cycles,
            field_mass_weighted_relative_l2=float(np.sqrt(err_p/max(den_p,1e-30))),
            piston_relative_l2=float(np.sqrt(err_x/max(den_x,1e-30))),
            midpoint_pressure_relative_l2=float(np.sqrt(err_mid/max(den_mid,1e-30))),
            function_evaluations=function_evaluations,
            initial_condition="zero state" if name=="startup" else "independently synthesized high-amplitude periodic snapshot"))
    save_json(directory/"report.json", dict(profile=ARGS.profile,
        historical_accuracy_claim=False, cases=reports,
        history_columns=["excitation_cycles","FEM_X","NN_X","FEM_pressure_mid","NN_pressure_mid"],
        reference="Full FEM RK4; states carried between blocks without resets",
        nn_integrator="DOP853", rtol=config["nn_rtol"], atol=config["nn_atol"],
        samples_per_case=cycles*per,
        error_definition="sqrt(sum((NN-FEM)^2)/sum(FEM^2)) over saved times; spatial pressure additionally uses lumped FEM mass weights; signed field error is FEM-NN"))
    log("predict_complete", cases=reports)


if __name__ == "__main__":
    config = configuration()
    stages = ("generate", "pod", "train", "predict") if ARGS.stage=="all" else (ARGS.stage,)
    for stage in stages:
        start = time.perf_counter()
        log(stage+"_started", profile=ARGS.profile, output=str(OUT))
        globals()[stage](config)
        elapsed = time.perf_counter()-start
        report_path = OUT/"stage_execution.json"
        execution = json.loads(report_path.read_text()) if report_path.exists() else {}
        execution[stage] = dict(profile=ARGS.profile, completed=True, elapsed_seconds=elapsed)
        save_json(report_path, execution)
        log(stage+"_finished", elapsed_seconds=elapsed)

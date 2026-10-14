# Pulsating sphere

This example reproduces the data-to-dynamics pipeline for a force-driven breathing sphere coupled to spherical acoustic radiation. It has four stages: FEM data generation, mass-weighted POD with physical amplitude augmentation, neural acceleration training, and independent time integration. Every generated file is written to the explicitly requested output directory. No saved project data or machine-specific source paths are required.

## Run

Use Python 3.10 or newer and install the two pinned dependencies in `requirements.txt` into your chosen environment. The verified dependency versions are NumPy 2.2.6 and SciPy 1.15.3. PyTorch, Mathematica, Matplotlib and Numba are not required.

```powershell
python -B run.py --profile smoke --output D:\Codex\Projects\acoustic\sphere_smoke --stage all
python -B run.py --profile paper --output D:\Codex\Projects\acoustic\sphere_paper --stage all
```

The default profile is `paper`; `--output` is always required. An absolute output path is recommended. The example also runs on Linux or macOS with a platform-appropriate output path. `TMP`, `TEMP`, `TMPDIR`, `MPLCONFIGDIR`, `NUMBA_CACHE_DIR` and the Python cache prefix are assigned below that output directory.

Each stage can be run separately with the same profile, configuration and output directory:

```powershell
python -B run.py --profile paper --output D:\Codex\Projects\acoustic\sphere_paper --stage generate
python -B run.py --profile paper --output D:\Codex\Projects\acoustic\sphere_paper --stage pod
python -B run.py --profile paper --output D:\Codex\Projects\acoustic\sphere_paper --stage train
python -B run.py --profile paper --output D:\Codex\Projects\acoustic\sphere_paper --stage predict
```

Use `--config PATH` for a separate JSON configuration. If an existing output directory has a different profile or pipeline configuration, every stage rejects the mismatch. Use a new directory for a different configuration. Rerunning a matching stage replaces that stage's own generated files. SHA-256 fingerprints of the actual numerical source files are computed at run time. A source mismatch is rejected. Stage proof files also verify the actual upstream arrays and model bytes before they are consumed.

## Physics and representation

The sphere has radius 0.05 m, effective mass 0.02 kg and uncoupled frequency 1000 Hz. Its damping ratio is 0.01. Air density and sound speed are 1.204 kg/m^3 and 343 m/s. The spring stiffness is M omega0^2 = 789568.352087 N/m, and the damping coefficient is 2 zeta M omega0 = 2.51327412287 N s/m. Fref = 0.001 N. These are benchmark parameters for an effective breathing coordinate, not a specified solid material model.

The continuous reference satisfies:

```text
M x'' + B x' + K x + 4 pi a^2 P = F(t)
P' + (c/a) P = rho c x''
p(r,t) = (a/r) P(t-(r-a)/c)
```

Data are generated from quadratic FEM for the acoustic potential, with psi=r phi and psi_tt=c^2 psi_rr. The normalized acoustic coordinate is y=psi/(a Vref lambda0); the normalized structural displacement is X=x/Xref. The FEM domain is z=(r-a)/lambda0 in [0,4], with exact outgoing-wave termination for spherical waves. The semidiscrete FEM time response is evaluated by an eigenvalue-based exponential-sum solution for the prescribed smooth forcing. Continuous analytical pressure is used only as a prediction reference.

POD acts on the joint structural-displacement/acoustic-potential coordinates. It is mass weighted, is not mean centered, and uses training snapshots only. The basis satisfies U.T M U = I. Modal coordinates are obtained as s.T M U. Pressure is reconstructed from the acoustic part of U qdot, not directly from U q. The modal network input is [q, qdot, F/Fref], and its output is qddot, with derivatives in tau=omega0 t.

## Profiles

| Setting | paper | smoke |
|---|---:|---:|
| Quadratic radial FEM elements | 64 | 8 |
| Full second-order coordinates | 130 | 18 |
| POD rank | 24 | 8 |
| Training duration in reference periods | 36 | 4 |
| Samples per reference period | 256 | 64 |
| Training frequency ratios | 0.90, 0.96, 1.04, 1.10 | 0.90, 1.10 |
| Training phases | 0, pi/2 | 0, pi/2 |
| Training amplitude ratios | 0.6, 1.2 | 0.6, 1.2 |
| Validation frequency ratios | 0.93, 1.07 | 1.00 |

The `paper` settings preserve the original training procedure. Eight unit-amplitude trajectories are scaled to two amplitudes, giving 16 training conditions; this is exact physical data augmentation for this linear FEM system. No random POD perturbations or synthetic nonlinear labels are added. After five-point fourth-order differentiation and sampling every fourth eligible time point, the paper dataset has 36864 training pairs and 4608 validation pairs. Its dimensions are 49 inputs and 24 accelerations.

The `smoke` profile checks that the whole pipeline runs with smaller data and fewer model-selection candidates. Its errors are not the manuscript's published accuracy. It uses the same physical constants and algorithmic structure with coarser spatial and temporal discretization.

## Training and stabilization

The network is 49-128-64-24 in the paper profile, with two Tanh hidden layers and a linear output layer. All biases are zero. Hidden weights are fixed random block features with seed 20260912; only output weights are fitted. Training inputs and acceleration labels are scaled by their training RMS, without mean subtraction. This example does not use Adam or backpropagation.

For the paper profile, feature scales 0.003 and 0.001 are tested against ridge ratios 0, 1e-14, 1e-12, 1e-10, 1e-8 and 1e-6. A nonzero ridge ratio multiplies the largest squared singular value of the hidden feature matrix. The original recorded selection was feature scale 0.001 and ridge ratio 1e-12; a fresh run selects by the maximum validation trajectory error, not by the analytical test pressure.

The training is **not pure least squares**. If the fitted zero-state companion Jacobian has positive growth exceeding 1e-7, an offline Schur decomposition separates its stable and unstable directions. A continuous algebraic Riccati solve, with state penalty 1e-10 I and identity control penalty, corrects the unstable subspace. The target acceleration Jacobian is translated back into the output weights using the pseudoinverse of the fixed-feature tangent. This correction uses the learned Jacobian; it does not insert the FEM or Galerkin coefficients into the network. The corrected output weights define the entire prediction model. `selection.json` and `selected.json` expose original growth, corrected growth, corrected dimensions and validation errors. This local correction is not a proof of stability for arbitrary trajectories.

## Independent prediction and outputs

`predict` runs one independent near-resonance case at f/f0=0.9744548193197631, amplitude ratio 0.9, onset constant 0.35 excitation periods and phase 0.63 rad. The frequency was selected from the continuous reference response; its analytical response is not a training label. The second-order neural model starts from 2r zero modal states and is integrated with DOP853 at rtol=2e-8, atol=1e-11 and maximum dimensionless step 0.2. States are never refilled from the reference trajectory.

The paper profile predicts 36 reference periods. This release intentionally provides the four-stage pipeline and one prediction case; it does not perform a frequency sweep, convergence study or continuation analysis.

Principal output files are:

- `generation.json`, `fem.npz`, `train_unit_*.npz`, `validation_*.npz`: generated FEM data and parameters.
- `pod.npz`, `pod.json`, `training_pairs.npz`: mass basis, retained fraction, shapes, orthogonality and training/validation pairs.
- `model.npz`, `selection.json`, `selected.json`: fixed features, corrected readout, RMS scales and actual selection records.
- `prediction.npz`, `prediction_metrics.json`: predicted and analytical nodal pressure fields, physical pressure in Pa, modal state, finite checks and relative L2 errors.
- `runtime.json`: executed stages, versions and numerical source fingerprints.

Point-pressure and radial weighted field errors measure different quantities. The generated one-case metrics are not substitutes for the manuscript's 50-case frequency-amplitude errors or its separate displayed space-time window metric.

## Recorded verification

Both profiles completed all four stages in a fresh Python 3.12 environment with the pinned NumPy/SciPy versions. The smoke run produced 512 training pairs, 64 validation pairs and a finite 257-by-17 pressure field; its mass-orthogonality error was 3.93e-16. Its large coarse-discretization prediction error is reported in `prediction_metrics.json` and is not claimed as manuscript accuracy.

The fresh paper run produced 36864 training pairs, 4608 validation pairs and a finite 9217-by-129 pressure field. The final clean-environment release run used two BLAS threads and selected epsilon=0.001 and ridge=1e-12. Its independent point-pressure relative L2 error was 0.00703430%, and its nodal radial weighted field error over z=0..4 and 36 reference periods was 0.01834932%. These are newly computed one-case metrics, not copied manuscript numbers. The original archived model selected ridge=1e-12 and reported a point error of 0.00947927%; nearly tied candidates can select differently after recomputation. The separate manuscript space-time-window and frequency-sweep metrics are not evaluated by this example. Both `--stage all` and `--stage generate` rejected a paper/smoke profile mismatch in an existing output directory.

# Two-dimensional Westervelt cavity: four-stage reproduction

This self-contained example implements full-order data generation, mass-weighted POD and offline augmentation, fixed-hidden neural-network output fitting, and autonomous neural-network RK4 prediction. It does not need the author's local project, a previously fitted 44-case NN, a frozen model hash, or machine-specific paths.

The **paper** profile uses **44 FEM/POD cases**: amplitudes [0.8, 0.9, 1.0, 1.1] and frequency ratios [0.95, 0.96, ..., 1.05]. Only **24 cases** enter the NN fit and RMS normalization: the same four amplitudes and ratios [0.95, 0.97, 0.99, 1.01, 1.03, 1.05]. The other 20 cases contributed to the shared 48-mode POD basis. They cannot be described as unseen cases of the entire pipeline.

## Install and run

The clean-environment release checks use 64-bit Python 3.12:

~~~powershell
python -m venv D:\Codex\Projects\acoustic\cavity_venv
D:\Codex\Projects\acoustic\cavity_venv\Scripts\python.exe -m pip install -r requirements.txt
D:\Codex\Projects\acoustic\cavity_venv\Scripts\python.exe -B run.py --profile smoke --output D:\Codex\Projects\acoustic\cavity_smoke --stage all
~~~

On another operating system, use its normal virtual-environment executable and choose a writable output path. All caches, temporary arrays and figures are placed under --output. Set this path to a D-drive directory on the author's Windows machine.

The smoke profile is a 233-node mesh, rank-4, two-case, short-horizon **workflow check**. Its errors and timings do not reproduce the paper. The paper profile is the default; its computation is substantial and is never launched by an import or a help command.

~~~powershell
D:\Codex\Projects\acoustic\cavity_venv\Scripts\python.exe -B run.py --profile paper --output D:\Codex\Projects\acoustic\cavity_paper --stage generate
D:\Codex\Projects\acoustic\cavity_venv\Scripts\python.exe -B run.py --profile paper --output D:\Codex\Projects\acoustic\cavity_paper --stage pod
D:\Codex\Projects\acoustic\cavity_venv\Scripts\python.exe -B run.py --profile paper --output D:\Codex\Projects\acoustic\cavity_paper --stage train
D:\Codex\Projects\acoustic\cavity_venv\Scripts\python.exe -B run.py --profile paper --output D:\Codex\Projects\acoustic\cavity_paper --stage predict
~~~

The equivalent complete run uses --stage all. The optional --config PATH accepts a JSON file with paper and smoke sections. An existing output directory must match both its stored effective configuration and actual source hashes; choose a new output directory after changing either. The original kernel requires ramp_cycles=30 and 17 perturbation levels spanning [-0.2, 0.2]; other values are rejected explicitly. Dependency versions are recorded at fit time. Completed per-case artifacts are reused only when recorded current-run identities and file hashes agree. Weighted POD construction checkpoints each completed case. An interrupted Gram or fit calculation restarts that calculation; completed case generation/projection/augmentation is retained.

## One full-resolution FEM example

To run only one representative FEM case, use the supplied `fem_single_case.json` configuration. It selects amplitude 1.0 and frequency ratio 1.0, with one worker and three FEM threads. The original 1024 × 512 grid, 1024 steps per period, zero initial state, forcing, and periodic convergence criterion remain unchanged.

From this example directory:

```bash
python -B run.py --profile paper --config fem_single_case.json --stage generate --output ./output_single_fem
```

This command generates one full-order case, 144 pressure snapshots, the final pressure/rate state, mesh, and completion records. It does not construct POD, augment data, train a network, or integrate an NN. The configuration is supplied for this `generate` command; use the original `config.json` for the full training workflow.

The actual full-grid check completed at **81 cycles (82,944 RK4 steps)**. Ten consecutive cycle differences satisfied the 0.1% threshold; the final difference was **0.0445651%**. The case timer recorded **220.98 s**, including integration, cycle comparisons, snapshot assembly, and case file writing, but excluding mesh assembly and parent-process setup. All saved fields were finite. See [the recorded single-FEM verification](../../docs/cavity_single_fem_verification_20261014.json). This verifies one FEM generation case, not the unexecuted full cavity training and NN prediction workflow.

A parent launcher can set ACOUSTIC_MAX_THREADS=N to cap per-process BLAS, FEM/Numba and augmentation threads; the effective values are saved in run_config.json. FEM worker count is separate: three paper workers with cap N can use up to 3N FEM threads. Direct runs retain configured thread limits.

## Numerical details

The physics.py module contains the connected Q1 mesh, exact reflection reduction about y=0.4, inlet forcing and coupled radiation, and the original optimized RK4 stages. The dataset.py module collects 144 pressure snapshots per POD case: 16 transient endpoints, 64 samples from the last cycle, and 64 additional transient phase samples. The POD uses full-domain-equivalent mass 2 * half_mass, balances each case by its weighted snapshot norm, retains an explicit constant direction, computes eigenvectors of the snapshot Gram matrix, and mass-orthogonalizes recovered modes. The paper profile builds the original 128 available directions and retains the first 48.

After POD construction, each NN-case FEM trajectory is reintegrated from zero. The exact-sample count is at most 10,000 per paper case, also bounded by the actual converged horizon sampled 128 times per period plus its endpoint. Labels are mass projections of the **actual full-order acceleration** captured along that trajectory, including a **native float64 final endpoint**. These differ from nonzero-level augmentation labels, whose nonlinear FEM RHS is recomputed at a reconstructed POD state. The trajectory's zero perturbation level keeps its original full-order label. Stored periodic fields, basis, offline operators and labels are float32; FEM evolving states, native endpoints, NN calculations and least-squares fitting are float64.

The paper fit selects 2,400 nonzero-level trajectory perturbations plus all 1,632 coordinate-axis rows per case: 96,768 rows overall. The original 44-case random-selection sequence is regenerated before selecting the 24 cases, so no old indices or old fitted model are required. Hidden layers are fixed; only the output matrix and bias are fitted. The conditioned hidden design is solved by dense SciPy gelsd with cutoff 1e-10 and then folded back into ordinary double-Tanh weights. No normal equations, streaming QR replacement, physical correction, PyTorch or GPU computation is used.

Prediction runs the four configured demonstration parameter pairs for 300 periods from the physical zero state. The learned NN acceleration alone supplies the RK4 derivative. A newly generated model.json supplies the model identity. The original 40 × 40 figure used FEM endpoint synchronization and a separate evaluation campaign; this example does not claim that its four demonstration trajectories regenerate that full figure automatically. Edit prediction_cases/prediction_cycles in a separate configuration and use a new output directory for another prediction scope.

## Resources and outputs

The resource_preflight.json gives current measured free memory/disk and formulas for retained storage and dense fitting scratch. For the paper configuration, the 96,768 × 4,749 float64 design alone is approximately 3.42 GiB. Its memory precheck reserves approximately 11 GiB for fitting; the check runs again immediately before fitting. Case-balanced weighted snapshots are approximately 9.73 GiB on disk. Additional snapshots, Gram, model and augmented labels are retained; free disk must exceed the estimate by 20%. No duration is predicted without measuring this exact configuration.

Important outputs are generated/, pod/phi.npy, exact/, augmented/, training/model.json, training/summary.json, prediction/*/trajectory.npz, and prediction/modal_pressure_norm.png. Shape, finite-value, disjoint-split, native reintegration, mass-orthogonality and output-folding checks fail explicitly. Prediction guard failures are recorded instead of reported as completed horizons. Public-source numerical settings match the experiment; byte-identical basis/model hashes are not promised across LAPACK/library versions or platforms.

See appendix_source.md for manuscript text and its source map. The recorded clean-environment workflow and cache-reuse checks are in smoke_verification.json; their timings describe only the coarse smoke profile.

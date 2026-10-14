# Reproducible acoustic dynamics examples

Three independent examples implement the same four-stage workflow:

**Generate reference data → construct the POD representation and augment training data → train the neural acceleration model → integrate the learned model.**

Each example starts from its numerical configuration. No private dataset, pretrained model, external notebook, or author-specific filesystem path is required.

| Example | Reference problem | Retained representation in the manuscript configuration |
| --- | --- | --- |
| [Pulsating sphere](examples/pulsating_sphere/README.md) | Coupled sphere and radial acoustic FEM | 24 mass-orthonormal modes of structural displacement and acoustic potential |
| [Piston and closed duct](examples/piston_duct/README.md) | Nonlinear piston coupled to a one-dimensional acoustic FEM | Piston coordinate plus 32 acoustic modes |
| [Westervelt cavity](examples/westervelt_cavity/README.md) | Two-dimensional neck–cavity FEM | 48 pressure modes; 44 POD cases and 24 NN training cases |

The inference stages advance generalized coordinates and velocities using the trained acceleration network. Reference solvers are used for generating labels and for the separately reported comparison. They do not correct the NN trajectory during integration.

## Installation

Use **64-bit CPython 3.12** and a new virtual environment. The dependencies in `requirements.txt` are pinned to the versions used for the clean-environment release checks.

```bash
git clone https://github.com/snaillpp-max/nonparametric-modeling.git
cd nonparametric-modeling
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install -r acoustic/requirements.txt
```

If PowerShell disables activation scripts, use `.venv\Scripts\python.exe` in place of `python`; activation is optional. No Mathematica or MATLAB installation is needed for these three Python examples.

## Run all four stages

First check the complete pipeline on a small numerical configuration:

```bash
python acoustic/run.py --example all --profile smoke --output acoustic/outputs/smoke --threads 2
```

On the authors' Windows workstation, outputs can instead be directed to a data drive:

```powershell
python acoustic/run.py --example all --profile smoke --output D:\acoustic-results\smoke --threads 2
```

`smoke` runs reference-data generation, POD/augmentation, actual NN fitting, and actual NN integration from an empty output directory. It is a software execution test at reduced numerical cost. **Its errors are not the manuscript errors and do not certify the manuscript mesh or POD accuracy.**

Run one example with its manuscript-scale configuration:

```bash
python acoustic/run.py --example pulsating_sphere --profile paper --output acoustic/outputs/paper --threads 2
python acoustic/run.py --example piston_duct --profile paper --output acoustic/outputs/paper --threads 2
python acoustic/run.py --example westervelt_cavity --profile paper --output acoustic/outputs/paper --threads 2
```

Read the individual README before a `paper` run. In particular, regenerating the cavity data on the 1024 × 512 background grid is an extended computation. A short smoke run cannot be used to estimate its total duration reliably. Use distinct output directories for different profiles.

The launcher runs examples sequentially, bounds BLAS/Numba thread counts, writes execution logs, and stops at the first failing example. Temporary files and numerical caches are kept inside the selected output directory.

## Run stages separately

The same example can be run one stage at a time. For example:

```bash
python acoustic/run.py --example pulsating_sphere --profile smoke --stage generate --output acoustic/outputs/stepwise
python acoustic/run.py --example pulsating_sphere --profile smoke --stage pod --output acoustic/outputs/stepwise
python acoustic/run.py --example pulsating_sphere --profile smoke --stage train --output acoustic/outputs/stepwise
python acoustic/run.py --example pulsating_sphere --profile smoke --stage predict --output acoustic/outputs/stepwise
```

Use the same profile and output directory for all four commands. The per-example documentation describes the saved arrays, model files, predicted trajectories, and error definitions. A nonzero exit status indicates a failed stage; retaining a partial output directory does not establish successful reproduction.

## Scope of reproduction

The release implements the four stages above. Periodic-orbit continuation, Floquet analysis, large parameter-plane campaigns, and figure-layout scripts are outside this release's scope. The manuscript settings and any differences between regenerated data and historical training data are described in [Appendices A–C](docs/appendices_ABC.md) and each example's README. Floating-point libraries, eigenvector signs, and regenerated training trajectories can change the fitted weights; successful execution is not a claim of identical historical model hashes.

The original `acoustic field.nb` is preserved as a historical notebook. It is not a dependency of the three examples and its older training settings must not be combined with the new Python configurations.

See the [release verification record](docs/verification.md) for the configurations actually executed, newly measured errors, timing scope, and the untested full cavity run. Automated Windows and Linux smoke checks are defined in the repository workflow.

For a smaller verification task at the full cavity mesh resolution, the [cavity README](examples/westervelt_cavity/README.md#one-full-resolution-fem-example) provides a single-case FEM-only command. That case has been executed through periodic convergence; it does not start the full data campaign or NN training.

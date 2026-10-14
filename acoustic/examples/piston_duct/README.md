# Nonlinear piston and linear acoustic duct

This example provides four stages: data generation, POD with label augmentation, fixed-hidden neural-network fitting, and independent integration. The network accepts 67 inputs and predicts 33 accelerations. The 32 acoustic modes and piston coordinate are retained in both profiles.

Install the repository requirements, then run from this directory:

```powershell
python -B run.py --profile smoke --output ./output_smoke --stage all
python -B run.py --profile paper --output ./output_paper --stage all
```

The default profile is `paper`. Each stage can also be called separately, in order, with the same profile, output path, and options:

```powershell
python -B run.py --profile paper --output ./output_paper --stage generate
python -B run.py --profile paper --output ./output_paper --stage pod
python -B run.py --profile paper --output ./output_paper --stage train
python -B run.py --profile paper --output ./output_paper --stage predict
```

The paper profile uses 2048 FEM elements, 2048 RK4 steps/cycle for the five baseline trajectories and 4096 for the eight amplitude-transition trajectories and prediction, 128 saved points/cycle, 64-cycle training trajectories, POD32, and 512 augmentation parents/case. Data generation stores about 1.8 GB of original state arrays; POD and training require additional memory and storage. The smoke profile uses 64 elements, 256 steps/cycle, 64 points/cycle, four-cycle trajectories, and 64 augmentation parents/case. **Smoke verifies execution and actual numerical errors; it does not establish manuscript accuracy.**

Prediction defaults to 64 cycles for paper and four for smoke. A 3000-cycle prediction can be requested when starting a new output:

```powershell
python -B run.py --profile paper --output ./output_3000 --cycles 3000 --stage all
```

This integrates both zero-start and high-amplitude initial conditions. Initial states and learned weights are regenerated; historical manuscript errors are not assumed. Expect materially more integration time for 3000 cycles.

Data generation synthesizes 26 outer periodic-response snapshot sets by independently solving fixed-frequency harmonic equations with the discrete FEM acoustic impedance. It also integrates 13 full FEM trajectories: startup, two prescribed frequency changes, two responses at frequency 48, and eight force-amplitude changes. Selection of the smallest/largest converged harmonic amplitude supplies data-generation initial conditions. No continuation or stability-analysis stage is implemented.

The historical frequency mesh and 39-case grouping are preserved. Periodic seeds are regenerated locally on the chosen mesh rather than importing the historical settled 1024-element initial-state files. Consequently the regenerated dataset, POD basis, and weights are not claimed bitwise identical to the manuscript model. See `appendix_source.md` for the verified historical settings and this distinction.

POD uses only training points in the original 31 cases. All labels, including unperturbed labels, are recomputed from lifted POD states through the complete FEM right-hand side. Augmentation parents belong exclusively to the training split; force values are preserved. The network uses two fixed Tanh hidden layers driven by piston displacement and a fitted affine connection from all inputs. Only its 3300 output/affine coefficients are fitted by SVD; prediction uses trained coefficients alone.

Artifacts:

- `config_used.json`, `provenance.json`: effective configuration and real source/config SHA256 hashes.
- `data/manifest.json`: case counts, finite checks, harmonic residuals, pressure-mean check.
- `pod/manifest.json`: split/augmentation counts, dimensions, mass orthogonality, FEM label agreement; plus basis and split arrays.
- `training/model.npz`, `training/report.json`: fitted weights, candidate selection, validation/test acceleration errors.
- `prediction/report.json`: actual full-field mass-weighted, piston, and midpoint-pressure relative L2 errors versus independently integrated full FEM.
- `prediction/*_history.npy`: columns are cycle, FEM X, NN X, FEM midpoint pressure, NN midpoint pressure. `*_last_cycles.npz` stores spatial fields.
- `stage_execution.json`: completed stages and measured elapsed times.

Relative L2 errors use the sum of squared prediction differences divided by the sum of squared FEM reference values over recorded samples, followed by a square root. Only the spatial pressure metric also uses lumped FEM mass weights. Zero-start and high-amplitude cases are reported separately.

NumPy, SciPy, and Numba are required. Temporary files and JIT caches are created under the selected output directory. In the current D-drive checkout, the default output and runtime caches remain on D. Choose an explicit D-drive output for Windows runs from another location. Reusing an output with different profile, effective settings, or source hashes is rejected.

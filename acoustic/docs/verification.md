# Verification of the public examples

The release was checked on 14 October 2026 using a fresh 64-bit Python 3.12.10 environment on Windows, with the pinned dependencies and two threads per process. Each complete run below started in an empty output directory and generated its own reference data, POD representation, augmented training data where applicable, model, and prediction. `pip check` reported no dependency conflicts.

| Example and profile | Actual scope | Measured complete invocation |
| --- | --- | ---: |
| Sphere, smoke | 8 elements, POD8, four reference periods | 1.19 s |
| Piston–duct, smoke | 64 elements, POD32, four prediction periods | 4.02 s |
| Cavity, smoke | 233 nodes, POD4, two training cases, two prediction periods | 9.64 s |
| Sphere, paper | 64 quadratic elements, POD24, one prediction case over 36 reference periods | 36.03 s |
| Piston–duct, paper | 2048 linear elements, POD32, 39 data cases, two predictions over 64 periods | 99.13 s |

These are measured local wall times for the complete invocation, including startup, numerical work, file I/O, and first-use compilation where applicable. They are not isolated NN inference times, are not the original manuscript timings, and are not portable runtime estimates.

The final sphere paper run produced 36,864 training pairs and 4,608 validation pairs. Its independent point-pressure relative L2 error was **0.00703430%**; the radial weighted pressure-field error was **0.01834932%**. Both cover the saved one-case prediction horizon. Nearly tied regularization candidates can select differently across BLAS versions and thread counts.

The piston paper run produced 109,824 raw samples and 35,880 augmented training samples, giving 112,736 fitting rows. Across 64 periods, full pressure mass-weighted relative L2 errors were **0.54811128%** for zero-start prediction and **0.17434405%** for the high-amplitude initialization. Each metric covers 8,192 recorded times and 2,049 spatial nodes. These are trajectory errors, not peak-amplitude or parameter-plane errors.

The cavity completed the reduced smoke pipeline and verified its data, model, and cache identities. The native FEM RHS, full-domain-equivalent mass, augmentation labels, fixed-hidden output folding, and paper training-index selection also received independent small-array/source checks.

A subsequent **single full-grid FEM case** used amplitude 1.0 and frequency ratio 1.0, retaining the original 1024 × 512 grid and 1024 steps per period. It completed 81 cycles (82,944 RK4 steps), with ten consecutive cycle differences below 0.1% and final difference 0.0445651%. All 144 × 206,081 pressure snapshots and the double-precision final pressure/rate states were finite. Its 220.98 s case timer covers integration, cycle comparisons, snapshot assembly and case output writing; it excludes mesh assembly and parent-process setup, so it has a different scope from the complete-invocation timings above. The [single-FEM report](cavity_single_fem_verification_20261014.json) records its actual configuration and output hashes. The example README provides its one-command invocation.

**The full cavity data campaign, POD/augmentation, training, and NN prediction were not rerun at paper scale.** Their total duration and regenerated prediction accuracy remain unmeasured here. The optional 3,000-period piston prediction and the sphere frequency sweep were also not part of these release checks.

Machine-readable results and numerical-source hashes are in [release_verification_20261014.json](release_verification_20261014.json). The [GitHub Actions workflow](../../.github/workflows/acoustic-smoke.yml) executes the four stages separately for each example on Windows and Linux, keeping its actual logs and outputs as temporary build artifacts. Consult the run status for the exact commit you use.

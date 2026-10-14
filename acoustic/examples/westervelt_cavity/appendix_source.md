# Appendix C. Two-dimensional Westervelt cavity reproduction

The domain joins a neck, [0, 0.25] × [0.35, 0.45], to a cavity, [0.25, 1] × [0, 0.8]. Pressure and pressure rate initially vanish. At the left neck mouth, ∂n(p + ηp_t) + p_t = A g(y) r(t/T) sin(ωt), where g(y) = √2 cos[π(y − 0.4)/0.1], r(c) = [1 − cos(π min(c/30, 1))]/2, β = 3, η = 5 × 10^−4, and ω = 50.418647180934734(f/f₀). Other boundaries are rigid. First-order inlet radiation is retained in the coupled acceleration.

The solver uses mass-lumped Q1 elements on a structured 1024 × 512 reference grid. Reflection about y = 0.4 leaves 206,081 computational nodes; the POD inner product uses twice the half-domain mass. Classical RK4 uses 1,024 steps per period. Zero-start trajectories span at least 67 periods and stop when the relative mass-weighted pressure difference between consecutive same-phase cycles remains below 10^−3 for ten cycles, with a 1,000-period limit.

The shared POD basis uses 44 parameter cases: A ∈ {0.8, 0.9, 1.0, 1.1} and eleven frequency ratios from 0.95 to 1.05, spaced by 0.01. Each case supplies 144 pressure snapshots: 16 transient endpoints, 64 samples from the last cycle, and 64 additional samples at eight phases of cycles 5, 10, 15, 20, 25, 30, 40 and 50. Mass weighting and per-case energy balancing precede snapshot-Gram eigendecomposition. An explicit constant direction is included, recovered directions are mass-orthogonalized, and 48 modes are retained.

NN fitting uses 24 cases, with the same amplitudes and frequency ratios {0.95, 0.97, 0.99, 1.01, 1.03, 1.05}. Actual FEM trajectories are reintegrated to obtain projected acceleration labels from at most 10,000 samples per case. Disjoint indices provide 1,776 augmentation bases, 1,500 validation states and 1,500 test states per case. Seventeen levels between −0.2 and 0.2 perturb modal pressure and rate while preserving instantaneous forcing. Nonzero-level targets recompute the nonlinear FEM RHS at reconstructed POD lifts; zero-level targets retain the actual full-order acceleration projection. Coordinate-axis anchors add 1,632 rows per case. Fitting selects 2,400 nonzero-level perturbations and all anchors, giving 96,768 rows.

The fixed-hidden double-Tanh network has dimensions 97–4748–4748–48, hidden scales 0.003 and 0.03, and second-layer diagonal gain 0.5. Only output weights and biases are fitted using double-precision LAPACK gelsd with cutoff 10^−10. Scaling uses selected training rows only. Autonomous prediction integrates the learned acceleration by zero-start RK4 without FEM corrections. Stored periodic fields and offline labels are single precision; evolving FEM states, native final labels and NN fitting are double precision. The 20 omitted NN frequencies contributed to POD and are not independent full-pipeline holdouts.

## Source map and public configuration

- Geometry, Q1 assembly and inlet profile: original convergence_half_domain.py, build_half and half_load; public physics.py.
- Coupled radiation, physical-zero generation and same-phase stopping: original fem_campaign.py; public physics.py and dataset.py, generate_case.
- Original RK4 kernels: convergence_stencil_rk.py, stage, and convergence_fused_finish.py, finish/cycle_fast; public physics.py.
- Shared 44-case, 144-snapshot mass POD: original rebuild_expanded_pod.py, build_basis; enrich_transient_snapshots.py, enrich; adopt_fixed_rank.py; public dataset.py, build_pod.
- Actual native trajectory labels: original reproject_expanded_exact.py, run; public dataset.py, project_exact.
- Perturbations and exact zero-level labels: original augmentation_revision_20260929/run_revision.py, augment_case; axis rows from prepare_new_basis_augmentation.py, append_axes; public dataset.py, augment_case.
- 24-case fit: original pure_NN/frequency_subset_trials/f095_step002_A080_A110_20261012/train_frequency_subset24.py; public network.py, selected_indices/weights/design/fold/train.
- Autonomous NN RHS and RK4: original pure_NN/long_prediction/run_long_prediction.py, nn_acc/drive/integrate_cycle; public network.py.
- All listed numerical settings appear in public config.json, paper section. The smoke profile intentionally changes mesh/rank/case counts/sampling/horizon; it validates execution only.
- Original NN model identity: fc77cbb8865946ad66904309f53a5781d1829ca35abf6e1d5553d47c336f756b. This is an archival reference, not an input or acceptance condition for new public results.

The manuscript prose describes the original study. The portable example regenerates its basis, arrays, model and dynamic provenance. Restructuring collects the same 144 snapshot phases during generation instead of making a separate transient pass, and removes the dependency on old rank-selection/mesh-audit reports. Those excluded audit campaigns and original error thresholds are not recreated or asserted. Tested public dependencies in requirements.txt differ from the author's previously recorded Python/NumPy/Numba runtime.

# Crunchy2 run: phase 4 scalar N150, 2M-grid certificate

This directory preserves the successful interval-arithmetic verification run
for `pinn_a_coeffs_N150_phase4_20260827.txt` and the subsequent additive
conversion to the half-DC cosine convention

```text
y(u) = a_0 / 2 + sum_{k=1}^N a_k cos(k u).
```

## Certified result

```text
candidate        = phase4_20260827_scalar_N150
coefficient count = 151
lambda           = 1.1687864806007486
beta             = 0.498
N_grid           = 2000000
Taylor order     = 6
Arb precision    = 128 bits
shard size       = 1024

bound midpoint   = +1.9928311263803702e-14
bound radius     = 1.5799937398052128e-13
bound interval   = [-1.3807106271671758e-13, +1.7792768524432498e-13]
bound ubound     = 1.7792768524432498e-13
D_lower lbound   = 3.4465883518530394e-01
```

The run log ends with `ALL_STAGES_COMPLETE`, and the final `s3` process has
exit status zero.

## Browse the run

The [expanded run directory](run/) preserves the original archive paths and
file contents, including the successful retry completed on August 29, 2026.

- [Run configuration and completion log](run/phase4_scalar_n150_20260827_2m_retry_run.log)
- [Stage logs, including all eight parallel workers](run/phase4_scalar_n150_20260827_2m_retry_logs/)
- [Final residual, denominator bound, and final-stage timing](run/phase4_scalar_n150_20260827_2m_retry_logs/s3.log)
- [Residual Arb dump](run/phase4_scalar_n150_20260827_2m_retry_work/staged_bound.arbdump)
- [Denominator lower-bound Arb dump](run/phase4_scalar_n150_20260827_2m_retry_work/staged_dlower.arbdump)
- [Original generated Lean certificate](run/phase4_scalar_n150_20260827_2m_retry_work/CertificatePhase4ScalarN150Phase420260827.lean)
- [Exact coefficient input](run/wave-cap-phase4-run/BabenkoRigorous/pinn_a_coeffs_N150_phase4_20260827.txt)
- [Archived Julia stage driver](run/wave-cap-phase4-run/BabenkoRigorous/scripts/stage_run_phase4_scalar_n150_20260827.jl)
- [Archived parallel cluster launcher](run/wave-cap-phase4-run/BabenkoRigorous/scripts/p1_parallel_run_phase4_scalar_n150_20260827.sh)

The run used eight parallel workers with four Julia threads each; the main
stages used eight threads. The physical parameters above enter the Julia
Babenko residual as `mu = lambda^2` and `beta_Babenko = beta^2`.

This is the archived completed run evidence, not the full cluster working
directory: the large per-grid intermediate shard arrays and the container
image are not included. The archived scripts retain their historical paths;
use the [Julia reproduction guide](../../README.md) with the current package
layout. Uploading these files did not rerun the numerical verification.

The original generated certificate is retained as historical output, before
the half-DC convention adjustment. For the publication statement, use the
half-DC certificate and audited entry points linked below. The Lean theorem
is conditional on certificate-specific numerical assumptions; these archived
outputs are not an independent Lean replay of the interval computation.

## Archives

`phase4_scalar_n150_20260827_2m_results_20260829.tgz` contains the raw run
log, all stage logs, the generated Lean certificate, the residual and
denominator Arb dumps, the coefficient file, and the dedicated run scripts.

`wave-cap-half-dc-alignment-20260830.tgz` contains the additive half-DC
conversion tools, test, Lean convention lemma, and alignment documentation.
No existing repository files were replaced when that archive was created or
extracted.

## SHA-256

```text
4d65357a64a0ccfbf0ed67f8dd0e1970f0ceb549a8cbf647d8490e4ac48abb07  phase4_scalar_n150_20260827_2m_results_20260829.tgz
ee8fe5db1ea3bbd6e37d0888fdadc00c9c9e1d6ce81cb9d46ae3e4db8ffcdf43  wave-cap-half-dc-alignment-20260830.tgz
```

## Publication Lean files

- [Half-DC certificate](../../../Lean/WaveCAP/PaperCertificate.lean)
- [Main theorem](../../../Lean/WaveCAP/PaperTheorems.lean)
- [Axiom audit](../../../Lean/WaveCAP/PaperAudit.lean)
- [Lean guide and numerical assumptions](../../../Lean/README.md)

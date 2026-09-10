# Phase 4 scalar N150, 2M-grid certificate

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

The half-DC Lean certificate and its audited entry points are stored in
`WaveCAP/WaveCAP/CertificatePhase4ScalarN150Phase420260827HalfDC.lean`,
`WaveCAP/WaveCAP/MainPhase4ScalarN150Phase420260827HalfDC.lean`, and
`WaveCAP/WaveCAP/AuditPhase4ScalarN150HalfDC.lean`.

# Half-DC certificate alignment

The Julia verifier and the current Lean definitions use different storage
conventions for the constant cosine mode:

- Julia: `y(u) = a[0] / 2 + sum(a[k] cos(k*u), k >= 1)`.
- Lean: `y(u) = sum(a[k] cos(k*u), k >= 0)`.

No existing file needs to change. A new Lean certificate should store
`a[0] / 2` as its zeroth coefficient and copy every positive-mode coefficient
unchanged. `WaveCAP/HalfDCConvention.lean` formalizes this adapter.

## Re-emit the completed two-million-point certificate

On `crunchy2`, from the `BabenkoRigorous` directory, run:

```bash
WORK=/scratch/ms14479/phase4_scalar_n150_20260827_2m_retry_work

apptainer exec --bind /scratch:/scratch \
  /home/tb97/transit-cap-ccf/cap-ccf.sif \
  julia --startup-file=no --compiled-modules=no --project=. \
  scripts/reemit_phase4_n150_half_dc_certificate.jl "$WORK"
```

The command reads the existing `staged_bound.arbdump` and
`staged_dlower.arbdump`; it does not rerun the interval calculation. It writes
the new file

```text
PaperCertificate.lean
```

inside the work directory. It refuses to replace that path if it already
exists. The original coefficient file, original certificate, and Arb dumps
remain unchanged.

The new module is named
`WaveCAP.PaperCertificate`. Its coefficient
vector has zeroth entry `-0.10288865038589269`, exactly one half of the source
value `-0.20577730077178538`; entries 1 through 150 are unchanged.

After downloading the new certificate, it can be integrated under its new
module name alongside the existing certificate. The certificate-specific
trust-anchor and headline modules should also receive new names rather than
replacing `CertificateAxioms.lean` or `Main.lean`.

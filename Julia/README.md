# Julia interval-arithmetic verifier

This directory is the `BabenkoRigorous` package. The package name and module
imports are unchanged. `src/` contains Fourier algebra, interval Taylor data,
certified Hilbert calculations, sharding, and residual assembly.

## Interval-certification strategy

The certificate input is a frozen even cosine polynomial in the Julia
convention $y(u)=a_0/2+\sum_{k=1}^N a_k\cos(ku)$. Its stored binary
coefficients and parameters become exact dyadic rational inputs to Arb.
The proof follows this chain:

1. **Finite Fourier algebra.** Construct $y$, $y_u$, and $H[y_u]$
   and differentiate or multiply their coefficient representations. Products
   retain their natural Fourier degree rather than being projected back to
   degree $N$. Arb balls enclose arithmetic roundoff.
2. **Nonlinear interval records.** Store values, derivatives and global
   derivative bounds at periodic grid nodes. Leibniz and composition rules
   propagate records through the capillary square root and reciprocal. A
   grid lower scan plus off-grid control proves $D>0$ before division.
3. **Certified nonlocal terms.** The capillary composite
   $G=(1+H[y_u])/D$ is not band limited. A sampled Hilbert transform
   receives an analytic aliasing correction based on derivative bounds.
   The implemented residual also routes $H[yy_u]$ through that path,
   although $yy_u$ is a finite polynomial. Higher-derivative Fourier-tail
   bounds support the transformed Taylor records.
4. **Continuum scan and export.** Assemble the residual record, apply an
   order-$M$ Taylor bound plus a rigorous remainder in each grid cell,
   and take the outward-rounded maximum. The denominator and residual
   endpoints are exported as exact rational Lean constants.

The large run is sharded: pointwise records stream to disk, Hilbert
transforms are processed by derivative column, and the final residual is
scanned shard by shard. Arb encloses numerical rounding; the aliasing and
off-grid corrections are explicit analytic estimates supplied by the code.
See the [Supplementary Methods](../docs/supplementary_computer_assisted_accuracy_proof.pdf)
for formulas and the [Lean guide](../Lean/README.md) for the numerical
trust boundary.

## Environment and tests

Use Julia **1.12.5**, the version recorded in `Manifest.toml`. From the
publication repository root:

```sh
julia --project=Julia -e 'using Pkg; Pkg.instantiate()'
julia --project=Julia Julia/test/runtests.jl
julia --project=Julia Julia/test/test_half_dc_certificate_additive.jl
julia --project=Julia Julia/test/test_sharded.jl
```

The last two test files are separate from the inherited `runtests.jl` entry
point and should be run explicitly. The manifest pins Arblib and the remaining
dependencies; no dependency versions were changed for this import.

## Publication candidate

The input is `pinn_a_coeffs_N150_phase4_20260827.txt`, with 151 coefficients in
the convention `y(u) = a0/2 + sum(a_k*cos(k*u), k=1:150)`.

The production driver
`scripts/stage_run_phase4_scalar_n150_20260827.jl` fixes Taylor order 6,
precision 128 bits, and `extra_orders=0`. It uses
`mu = 1.1687864806007486^2` and `beta = 0.498^2` in the Babenko residual.

Archived logs, Arb endpoints, original run scripts, and certificate-conversion
evidence are in
[`results/phase4_scalar_n150_20260827_2m_20260829/`](results/phase4_scalar_n150_20260827_2m_20260829/README.md).
The original archive is also expanded there into a browsable `run/` directory,
with links to individual logs, scalar dumps, inputs, and scripts.

## Staged reproduction without a cluster-specific launcher

From the publication repository root, the following runs the grid stages
sequentially with ordinary Julia. The work directory is absolute so it remains
unambiguous if the working directory changes. This is the full computation,
not a small test.

```sh
julia --project=Julia Julia/scripts/stage_run_phase4_scalar_n150_20260827.jl p1 "$PWD/runs/phase4_n150" 1024 2000000
julia --project=Julia Julia/scripts/stage_run_phase4_scalar_n150_20260827.jl fft "$PWD/runs/phase4_n150" 1024 2000000
julia --project=Julia Julia/scripts/stage_run_phase4_scalar_n150_20260827.jl s3 "$PWD/runs/phase4_n150" 1024 2000000
julia --project=Julia Julia/scripts/reemit_phase4_n150_half_dc_certificate.jl "$PWD/runs/phase4_n150"
```

The stage driver retains its original certificate output convention. Use the
last command to produce the half-DC-adjusted certificate for comparison with
the dedicated Lean certificate. It writes a new file in the run directory;
it does not overwrite the publication certificate.

For parallel cluster execution,
`scripts/p1_parallel_run_phase4_scalar_n150_20260827.sh` preserves the original
Apptainer launcher. Configure `SIF`, `RUNTIME`, and `BINDPATH` for the target
machine. Its original default image path is specific to the source cluster,
and `/usr/bin/time -v` expects a GNU/Linux environment.

## Source organization

- `src/FourierSeries.jl`: finite Fourier differentiation, Hilbert action, and bounds.
- `src/TaylorSeries.jl`: interval derivative records, products, tightening, and scans.
- `src/ResidualBound.jl`: monolithic residual pipeline.
- `src/Sharded.jl` and `src/ShardStore.jl`: staged, bounded-memory execution.
- `src/Certificate.jl`: certificate text generation.
- `scripts/half_dc_certificate_tools.jl`: additive constant-mode conversion.

The archived two-million-point result is a full residual upper bound of
approximately `1.7792768524432498e-13`. Its interpretation and numerical
assumptions are described in the repository README and mathematical note.

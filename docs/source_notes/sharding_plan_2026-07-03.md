# Sharding plan for BabenkoRigorous — RAM reduction (2026-07-03)

Goal: run `residual_bound` at large `N_grid` without holding ~20 full
`TaylorSeries` matrices (`N_grid × (order+1)` Arbs each) in RAM
simultaneously. Julia-only change; the Lean side is untouched (see §6).

## 1. Data-flow analysis of `_residual_bound_full`

Every step of the pipeline falls into one of three classes:

**(a) Pointwise in `u` — recomputable per shard from O(1) data.**
The trig chain is exact and coefficient-level: `y`, `yu`, `Hyu` and all
their derivatives are `TrigPoly`s (~110 coefficients), evaluable anywhere
via `tp_eval`. Everything built from them by per-point recurrences is
pointwise: `trigpoly_to_taylor` rows, Leibniz products (`ts_y_Hyu`,
`ts_y_yu`, `ts_yuD`, `ts_G = B·(1/D)`), `sqrt_series`/`inv_series`
(Faà di Bruno per point), `deriv_series` (index shift), the B3 matrix
(ResidualBound.jl:167–175), and the final linear assembly of `ts_R`.
None of these matrices needs to exist globally.

**(b) Grid reductions — streamable.** `dmin_bound`'s grid minimum,
`sup_bound`'s grid maximum (max over `j` of the per-point Taylor sum,
TaylorSeries.jl:158–170), and the `tighten_sup_bounds!` grid+Lipschitz
damping are all associative min/max reductions over grid points: process
shard-locally, combine across shards. All `sup_bounds` vectors are O(M)
scalars.

**(c) Global — the only non-local steps.** The two `fft_hilbert_series`
applications (`H[y·y_u]` at Step 4, `H[∂_u G]` at Step 6c). Each is
`M_order+1` independent length-`N` certified convolutions
(`fft_hilbert`: AcbPoly multiplication of the input column against the
cot-kernel, FFTHilbert.jl:111–190). Each convolution needs one full
column at once — but only one at a time.

**Output requirement.** `emit_certificate` consumes exactly two scalars:
`bound` and `D_lower` (Certificate.jl:186–200). The NamedTuple that pins
`ts_R, ts_y, ts_yu, ts_Hyu, ts_G3` in RAM is not needed for
certification.

## 2. Sharded architecture

Grid split into `S` contiguous shards of `P = N_grid / S` points
(choose `P ≈ 4096–16384`). Four phases, strictly sequential:

**S0 — coefficient-level scalars (RAM: KBs).** Build the `TrigPoly`
derivative towers for `y`, `yu`, `Hyu` up to order `M3+1`; closed-form
`supnorm_bound`s; `tight_DG_sup_bounds`; grid constants.

**S1 — streaming reductions (RAM: one shard).** One pass over shards:
per shard, evaluate the pointwise chains needed by the scalar
bookkeeping (D-chain for the grid min; per-point values feeding
`tighten_sup_bounds!`-style maxima). Outputs: `D_lower`, all tightened
`sup_bounds` vectors, `Gpp/Gppp` bounds. All scalars. (Per-point data
discarded; the chains are recomputed later — see cost note in §4.)

**S2 — sequential global transforms (RAM: ~4 columns + poly buffers).**
For each of the two Hilbert applications, for each level
`m = 0..M_order`:
1. stream-generate the full input column ((y·y_u)^{(m)} resp.
   G^{(m+1)} at every grid point) by pointwise recompute, shard by
   shard, into one preallocated `Vector{Arb}(N)`;
2. run `fft_hilbert` on it (reuse ONE cot-kernel vector and ONE set of
   AcbPoly buffers across all `2(M_order+1)` levels — the kernel is
   level-independent; currently it is rebuilt per call);
3. write the output column to per-shard files (§3); free.

Peak: input col + kernel + output col (3N Julia Arbs) + AcbPoly buffers
(4N Acb coefficients, C-allocated, no per-element Julia overhead).

**S3 — shard-by-shard residual assembly (RAM: one shard × full chain).**
For each shard: read the `2(M_order+1)` stored H-column slices;
recompute all pointwise chains for the shard's points; assemble the
shard's `ts_R` rows; update the running `sup_bound` maximum (per-point
Taylor sum + scalar remainder radius, exactly the current formula);
free. Return `(bound, D_lower)` — scalars only.

## 3. Shard store & exact serialization

- `ShardStore`: a directory + manifest (`N_grid, P, prec, M_order`,
  column names, shard checksums).
- Balls must round-trip **exactly** to preserve rigor: serialize with
  `Arblib.dump_string` / `Arblib.load_string!` (wrapping
  `arb_dump_str`/`arb_load_str` — exact hex mantissa/exponent + radius).
  Do NOT go through decimal printing.
- Optional: also cache the D/G-chain columns (~order M3) to trade disk
  for the S1/S2/S3 recompute (§4).

## 4. RAM estimate

Bookkeeping unit: one "column" = `N_grid` Arbs. At `prec=128` an Arb is
a 48-byte struct + Julia object header + pointer slot + finalizer
overhead ≈ **~100 B effective**; AcbPoly coefficients are C-inline
≈ 96 B with no per-element overhead.

**Current live set** (β≠0, M_order=6, extra_orders=e): counting the
matrices in `_residual_bound_full` — y_h/yu_h/y_yu_h at order M+2+e,
D3/invD3/B3/G3/yu_M3 at order M+3+e, plus ~10 matrices at orders
M..M+1 — gives **~170 columns at e=0, ~250 at e=8** (GC reclaims some
mid-pipeline; the return NamedTuple alone pins ~38–60). Estimates:

| N_grid | current live (~170–250 cols × 100 B) | observed-RSS ballpark |
|--------|--------------------------------------|------------------------|
| 65 536 | 1.1–1.6 GB | 2–3 GB |
| 262 144 | 4.5–6.5 GB | 8–13 GB |
| 1 000 000 | 17–25 GB | 30–50 GB |

**Sharded peak** = max(S2 transient, S3 shard):
- S2: 3 Arb columns (~300 B·N) + 4N Acb poly coefficients (~384 B·N)
  ≈ **~0.7 KB per grid point** → 45 MB at N=65536; 0.7 GB at N=10^6.
- S3: P × (~170–250 values/point) × 100 B → **50–200 MB** for
  P = 4096–8192, independent of N.

| N_grid | current | sharded peak | saving |
|--------|---------|--------------|--------|
| 65 536 | ~2–3 GB | ~0.1–0.2 GB | ~15–25× |
| 262 144 | ~8–13 GB | ~0.25 GB | ~35–50× |
| 1 000 000 | ~30–50 GB | ~0.8 GB (S2-dominated) | ~40–60× |

The grid-proportional coefficient drops from ~170–250 columns to ~7
column-equivalents (S2), and everything else becomes N-independent.

**Disk**: stored H-columns = `2(M_order+1)` = 14 columns × N ×
~60–100 B serialized ≈ 90 MB at N=65536, ~1.4 GB at N=10^6. Optional
D/G-chain cache adds ~25 columns (~2.5 GB at N=10^6).

**CPU**: without the optional cache, pointwise chains are evaluated
~3× (S1, S2 input generation, S3) instead of once. The chains are
cheap (trig eval + Bell-polynomial recurrences, `Threads.@threads`
within shard/column); expect ≤2× wall-clock vs current at equal N.
The convolutions (the actual FLOP hot spot) are unchanged.

## 5. Optional stage 2 — overlap-add convolution

If the ~0.7 KB/point S2 transient itself becomes the constraint
(N ≥ 10^7), replace the single length-N AcbPoly multiplication with
overlap-add block convolution: split both the column and the kernel
into blocks of length B, accumulate the (N/B)² block products. Memory
drops to O(B); certified error tracking is unchanged (ball arithmetic
per block product; the fold in FFTHilbert.jl Step 3 generalizes).
Cost: ~(N/B)× more transform work. This makes the whole pipeline's
peak RAM **N-independent** (~100–200 MB, S3-dominated).

## 6. Rigor invariants

1. **Exact serialization** (§3) — shards are bit-exact balls.
2. **Reduction order**: keep ascending-`j` order within and across
   shards so ubound-comparison tie-breaking matches the sequential
   loops (`sup_bound`, `dmin_bound`) and results stay **bit-identical**
   to the monolithic path.
3. **Validation**: run monolithic vs sharded at N=4096 (β=0 and β≠0)
   and assert `Arblib.isequal` on `bound`, `D_lower`, and every stored
   column — bit-identical, not just overlapping.
4. **Lean side: zero changes.** The per-primitive statements are
   theorems universally quantified over the witness (P2–P6,
   `audit_2026-07-03.md`), and the two numeric axioms
   (`cert_arb_dmin_bound`, `cert_arb_sup_bound_le_eps0`) only need the
   same two scalars to come out. Sharding is invisible to the trust
   chain.
5. Interplay with Q1 (stale emitter): the sharded driver should return
   `(bound, D_lower)` scalars and feed the fixed emitter; do Q1 first
   or together.

## 7. Implementation checklist

1. `ShardStore` module (manifest, dump/load via `Arblib.dump_string`).
2. Refactor pointwise kernels to take a `j`-range + preallocated
   `P×(order+1)` buffer: `trigpoly_to_taylor!`, Leibniz `mul!`,
   `sqrt_series!`/`inv_series!` per-point bodies, B3 evaluation.
   Full-grid wrappers = one-shard calls (existing tests stay valid).
3. Split reductions: shard-local `sup_bound`/`dmin_bound`/tighten
   kernels + cross-shard combiner (ascending order).
4. S2 driver: column generator → `fft_hilbert` (hoist kernel + AcbPoly
   buffer allocation out of the loop) → shard writer.
5. S3 driver: shard reader + per-shard R assembly + running max.
6. New entry point `residual_bound_sharded(...; shard_size, workdir)`
   returning `(bound, D_lower)`; keep `residual_bound` for small N and
   for the validation cross-check.
7. Validation test (§6.3) added to `test/`.

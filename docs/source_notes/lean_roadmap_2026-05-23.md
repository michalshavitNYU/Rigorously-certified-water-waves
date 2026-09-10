# Lean roadmap — deriving `cert_residual_le` instead of axiomatizing it

> **STATUS UPDATE (2026-05-24): All six phases complete.** The trust
> footprint is at its minimum — 3 Mathlib standard axioms +
> 7 atomic `cert_arb_*` axioms (one per Arb primitive used in the pipeline).
> The original monolithic `cert_residual_le` and `cert_D_ge` are now derived
> theorems. See [`status_2026-05-24.md`](status_2026-05-24.md) for the
> current snapshot.

**Date:** 2026-05-23
**Driver:** today's Julia tightening work added new algorithmic pieces
(`tighten_sup_bounds!`, `tight_DG_sup_bounds`, TrigPoly `*`, `dmin_bound`
max-of-two-bounds selection) that have no Lean justification yet. More
broadly, the WaveCAP Lean tree currently has the universal analytic
lemmas (N1–N5) but **no per-operation correctness lemmas** for the
TaylorSeries / TrigPoly / FFT-Hilbert algorithms the Julia residual
pipeline uses. The trust anchor `cert_residual_le` therefore axiomatises
the entire residual computation, not just the irreducible Arb-primitive
trust.

This doc lays out the architecture and roadmap for closing that gap.

---

## 1. Architecture goal

For every Julia algorithm that produces an Arb output with a claimed
semantic (e.g., "this Arb's upper bound is ≥ `‖f^(m)‖∞`"), there should
be a **Lean lemma** that proves the claim **assuming the Arb primitives
(add, mul, sqrt, cos, sin) are rigorous** — i.e., assuming each Arb
input/output ball really does contain the true real value, the algorithm's
output Arb really does contain the true output semantic.

The final theorem `‖R‖∞ ≤ ε₀_cert` then becomes a **derived** theorem:

```
‖R‖∞ ≤ ε₀_cert
  := apply (sup_bound_taylorseries_correct) to ts_R,
     where ts_R is established by composition of all the per-operation
     lemmas above, each fed an Arb input whose correctness is taken as
     primitive (small atomic `cert_*` axioms — one per *Arb primitive
     correctness claim*, NOT one per intermediate quantity).
```

The trust footprint shrinks from "one giant axiom on the full residual"
to "Arblib/FLINT correctness + Lean kernel + Mathlib", which is the
trust footprint that paper-level CAPs aim for.

---

## 2. Inventory of missing pieces

Cross-reference: [changes_2026-05-22_midpoint_convention.md](changes_2026-05-22_midpoint_convention.md)
and the [Julia source](../BabenkoRigorous/src/).

| Area | Julia source | Lean status (2026-05-24) | File |
|---|---|---|---|
| **A. TaylorSeries struct semantics + arithmetic** | `TaylorSeries.jl` (struct + `+`, `-`, scalar `*`, `*` Leibniz, `deriv_series`, `truncate_order`, `trigpoly_to_taylor`) | ✅ done — `IsTaylorSeriesOf` + 9 correctness lemmas, axiom-clean | `TaylorSeriesCorrectness.lean` |
| **B. `sqrt_series`, `inv_series` per-grid recursion** | `TaylorSeries.jl:510–574` | ✅ done — `sqrt_series_correct`, `inv_series_correct`, axiom-clean | `SeriesCorrectness.lean` |
| **C. `tighten_sup_bounds!` recursion** | `TaylorSeries.jl:259–290` | ✅ done — MVT-based step + downward induction + bridge | `TightenSupBoundsLemma.lean` |
| **D. `tight_DG_sup_bounds` algebraic recurrences** | `TaylorSeries.jl:752–847`, `FourierSeries.jl:240` | ✅ done — `D² = A²+B²`, `G·D = B`, Leibniz expansions, sup-bound recurrences, `trigpoly_supnorm_bound` | `TightDGSupBounds.lean` |
| **E. `dmin_bound` correctness** | `TaylorSeries.jl:655–751` | ✅ done — `D_lipschitz_bound`, `d_lower_init_correct`, `max_of_two_bounds_correct`, `dmin_bound_correct` | `DminBoundCorrectness.lean` |
| **F. `fft_hilbert` + `fft_hilbert_series`** | `FFTHilbert.jl` | ✅ done — Poisson-summation aliasing bound, sampling-DFT vs cot-convolution equivalence, `IsTaylorSeriesOf` for `H[G]` | `FFTHilbertCorrectness.lean`, `AliasingErrorBound.lean` |
| **G. Residual assembly + final bound** | `ResidualBound.jl` | ✅ done — full composition of A+B+C+D+E+F over the residual chain → `cert_residual_le_derived` | `ResidualAssemblyCorrectness.lean` |
| **Auxiliary — Hilbert smoothness** | (not in Julia; Mathlib gap) | ✅ done — `hilbert_contDiff` at the tight `+1` loss, via Cauchy-Schwarz on Fourier coefficients | `HilbertSmoothness.lean`, `HilbertSharper.lean` |
| **Auxiliary — pipeline smoothness helpers** | (not in Julia) | ✅ done — `D_deriv_eq` (chain rule), `yu_deriv_bound`, `Hyu_deriv_bound`, periodicity helpers, ContDiff for trig polys | `PipelineSmoothness.lean` |
| **Auxiliary — residual smoothness** | (not in Julia) | ✅ done — `R_contDiff` for the full Babenko residual given `D > 0` | `RSmoothness.lean` |
| **Phase 6 — atomic axioms** | (Julia primitives' Arb correctness) | ✅ done — 7 atomic `cert_arb_*` axioms; old monoliths now derived theorems | `CertificateAxioms.lean` |

---

## 3. Dependencies

```
   A (TaylorSeries semantics + arithmetic)   ← foundational
       │
       ├─→ B (sqrt/inv per-grid recursion)
       │       │
       │       └─→ E (dmin_bound)
       │              │
       │              └─→ G (residual assembly via ts_D, ts_invD3, ts_G3)
       │
       ├─→ C (tighten_sup_bounds!)           ← orthogonal to B
       │       │
       │       └─→ E, F, G (consumers benefit from tighter sup_bounds)
       │
       ├─→ D (tight_DG_sup_bounds)           ← orthogonal to B
       │       │
       │       └─→ G (G^(m+3), G^(m+4) bounds for fft_hilbert_series)
       │
       └─→ F (fft_hilbert + fft_hilbert_series)
              │
              └─→ G (H[y·y_u], H[∂_u G] are inputs to ts_R)
```

A is the prerequisite for everything else. B is the prerequisite for E,
and downstream for G. C, D, F all depend only on A.

---

## 4. Phased roadmap

### Phase 1 — Foundation (A) — ✅ done
`TaylorSeriesCorrectness.lean`. Defines `IsTaylorSeriesOf ts f` and proves
one correctness lemma per arithmetic operation.
**Final size:** ~720 lines, 9 theorems, axiom-clean.

### Phase 2 — Quick wins (C, D) — ✅ done
- `TightenSupBoundsLemma.lean` (C). MVT-based step lemma plus downward
  induction; bridge to `TaylorSeriesData`. Final: ~875 lines.
- `TightDGSupBounds.lean` (D). Algebraic identities, Leibniz expansions
  for `D·D` and `G·D`, sup-bound recurrences, `trigpoly_supnorm_bound`.
  Final: ~860 lines.

### Phase 3 — Recursive series (B) — ✅ done
`SeriesCorrectness.lean` for `sqrt_series` and `inv_series`. Per-grid
recurrence via induction; sup_bounds parameterized by `Φ` (caller supplies
closed-form bound on `|sqrt^(k)|` or `|inv^(k)|`). Final: ~290 lines.

### Phase 4 — Spectral pieces (F) — ✅ done
`FFTHilbertCorrectness.lean`. Wraps `AliasingErrorBound` (N4, deeply
rebuilt via Poisson summation) and `FourierTailHilbert` (N3) into
`IsTaylorSeriesOf` for `H[G]`. After Phase 5 rewiring, the
`h_H_contdiff` hypothesis is discharged internally via `hilbert_contDiff`.
Final: ~470 lines.

### Phase 5 — Closing the loop (E + G) — ✅ done
- `DminBoundCorrectness.lean` (E). After Phase 5 rewiring, the chain rule
  and derivative bounds are discharged internally via `PipelineSmoothness`.
  Final: ~340 lines.
- `ResidualAssemblyCorrectness.lean` (G). Full composition over the
  residual chain. Headline `cert_residual_le_derived` parameterized by
  the per-step Arb-trust bundle. Final: ~800 lines.

### Phase 6 — Audit trust footprint — ✅ done
`CertificateAxioms.lean` decomposes the original two monolithic axioms
(`cert_residual_le`, `cert_D_ge`) into **7 atomic `cert_arb_*` axioms**,
one per Arb primitive used in the pipeline. The old "axioms" are now
derived theorems. Final trust footprint:

```
[propext, Classical.choice, Quot.sound,
 cert_arb_dmin_bound,
 cert_arb_fft_hilbert_series,
 cert_arb_inv_series,
 cert_arb_sup_bound_le_eps0,
 cert_arb_trigpoly_to_taylor,
 cert_arb_ts_deriv,
 cert_arb_ts_mul]
```

The trust trade is now exactly as planned: trust each FLINT/Arblib
primitive separately, not the entire residual algorithm as a whole.

### Auxiliary work (not on the original roadmap but landed)

- **`HilbertSmoothness.lean` + `HilbertSharper.lean`** — proves the
  Mathlib gap "Hilbert preserves smoothness on periodic functions" at
  the tight `+1` loss (via Cauchy-Schwarz on Fourier coefficients +
  Parseval). Eliminates the `h_H_contdiff` caller burden in F and G.
- **`PipelineSmoothness.lean`** — collects derivative bounds for `y_u`,
  `H[y_u]`, the chain rule for `D = sqrt(...)`, periodicity helpers,
  and `C^∞` for trig polys. Eliminates 7 hypotheses from E's signatures.
- **`RSmoothness.lean`** — proves `ContDiff k (R a μ β)` for any `k`
  given `D_lower > 0`. Enables the Phase 6 derivation of
  `cert_arb_residual_TS` from the per-primitive axioms.

---

## 5. Today's immediate work

Drafting **theorem statements only** (no proofs, just `sorry`) for:

- (A) `TaylorSeriesCorrectness.lean` — see [lean_taylorseries_correctness_plan.md](lean_taylorseries_correctness_plan.md)
- (C) `TightenSupBoundsLemma.lean` — see [lean_tighten_sup_bounds_plan.md](lean_tighten_sup_bounds_plan.md)
- (D) `TightDGSupBounds.lean` — see [lean_tight_dg_sup_bounds_plan.md](lean_tight_dg_sup_bounds_plan.md)

After signatures are agreed, proofs follow.

---

## 6. Why this isn't just "we already have N1–N5"

The existing N1–N5 lemmas are **universal analytic facts** (e.g., "Faà di
Bruno applied to `|·|` gives a sup-bound", "DFT aliasing decays as
`N^{-(p-1)}`"). They are necessary but not sufficient. To certify the
specific Julia pipeline, we need lemmas that *connect each Arb data
structure to a real-valued function* — i.e., the `IsTaylorSeriesOf`
predicate at the heart of (A), and then one application of an N1–N5
lemma per algorithmic step.

Without (A)–(G), the Lean side has the *facts* the algorithm relies on
but doesn't actually *certify the algorithm*. The trust anchor
`cert_residual_le` papers over that gap.

---

## 7. Cross-references

- [overview.md](overview.md) §6 — current Lean component description.
- [lean_design.md](lean_design.md) — per-file design for existing files.
- [milestones.md](milestones.md) — M8–M13 cover existing Lean milestones.
  This roadmap defines new milestones M14–M19 for the missing pieces:
  - M14: TaylorSeriesCorrectness (A)
  - M15: TightenSupBounds (C)
  - M16: TightDGSupBounds (D)
  - M17: SeriesCorrectness (B)
  - M18: FFTHilbertCorrectness (F)
  - M19: DminBound + ResidualAssembly (E + G)
- [changes_2026-05-22_midpoint_convention.md](changes_2026-05-22_midpoint_convention.md) §6 — the original Lean audit that
  prompted this roadmap.

# Bug fix — restore the cap-ccf `add_error!` convention in TaylorSeries

**Date:** 2026-05-22
**Files touched:** `BabenkoRigorous/src/TaylorSeries.jl` only.
**Status:** restores the convention that was supposed to apply from M2 onward,
ported from cap-ccf. **Not a new design** — undoing a regression.
**Driver / verification:** `BabenkoRigorous/scripts/run_pinn.jl` against the
PINN-projected coefficient set `pinn_a_coeffs_N110.txt`.

## 1. The intended convention (from cap-ccf)

The project convention — established in cap-ccf and meant to apply throughout
`BabenkoRigorous` — is:

> **`sup_bounds`-derived corrections (Taylor remainder, Lipschitz, aliasing)
> enter the result's RADIUS via `Arblib.add_error!`, never the midpoint via
> plain Arb addition/subtraction.**

This keeps the invariant `midpoint = best estimate of the value,
radius = rigorous looseness`. The cap-ccf reference implementations:

- [`cap-ccf/.../taylor_series.jl:298`](../../../CAP/cap-ccf/pkg/CCFBlowup.jl/src/taylor_series.jl):
  `return add_error(taylor_sum, abs(remainder))` — Taylor evaluation with
  remainder ball.
- [`cap-ccf/.../taylor_series.jl:382`](../../../CAP/cap-ccf/pkg/CCFBlowup.jl/src/taylor_series.jl):
  `Arblib.add_error!(s, rem_radius)` — `shift_taylor`.

`FFTHilbert.jl:191` in this repo already follows the convention:
`Arblib.add_error!(HG_vals[j], ε_alias)`.

`TaylorSeries.jl`'s `sup_bound`, `inf_bound`, and `dmin_bound`'s Lipschitz
step **did not** follow the convention — they folded the bound contribution
into the midpoint via plain Arb `+`/`−`. The symptom: `dmin_bound` returned
`D_lower` with midpoint = `-196` for the PINN input, even though the true
`min D ≈ 0.34` and per-grid Arb evaluations of `D(u_j)` had midpoints that
matched Float64 to 10⁻¹². This is what surfaced the regression.

## 2. Verification after the fix

`run_pinn.jl` at `M_order = 4`, `N_grid = 2048`, `prec = 128`:

| Quantity | Arb midpoint | Float64 truth | rel. err |
|---|---:|---:|---:|
| Per-grid `D(u_j)` | `[0.3447, 1.8532]` | `[0.3447, 1.8532]` | 10⁻¹² |
| `dmin_bound` returned `D_lower` | 0.3445 | 0.3447 (min D) | 5 × 10⁻⁴ |
| `residual_bound` midpoint, β = 0 | 1.693 | 1.691 | 8 × 10⁻⁴ |
| `residual_bound` midpoint, full PINN | 1.576 × 10⁻⁷ | 1.518 × 10⁻⁷ | 0.038 |

All midpoints now agree with the Float64/FFT truth.

Rigorous `lbound`/`ubound` semantics are unchanged (mathematically identical
under either convention; the change is only where the looseness is recorded
in the Arb representation).

## 3. Code audit summary

Grepped all of `BabenkoRigorous/src/` for patterns that combine a `sup_bound`-
derived quantity with per-grid midpoint data. Three regressions, all in
`TaylorSeries.jl`, no others:

| Function | Status before | Fixed? |
|---|---|---|
| `sup_bound` | `s = s + Fbound·rem_factor` | ✅ |
| `inf_bound` | `lb = f0 − (taylor_tail + Fbound·rem_factor)` | ✅ |
| `dmin_bound` Lipschitz step | `Arb(D_lb_min_arf) − lip_correction·h` | ✅ |
| `dmin_bound` return value | always `inf_bound(ts_D)` | ✅ (see §4.4) |
| `FFTHilbert.fft_hilbert` aliasing | already `add_error!` | n/a (correct) |
| `TaylorSeries` `*`, `+`, scalar `*`, `deriv_series`, `truncate_order` | per-grid arithmetic only; no sup_bound→midpoint mixing | n/a (correct) |
| `inv_series`, `sqrt_series` per-grid recursion | uses `ts.derivs` via closed-form, no sup_bound mixing | n/a (correct) |
| `_inv_sup_bounds_on_interval`, `_sqrt_sup_bounds_on_interval`, `_faa_di_bruno_derivs` | pure bound-only constructions; no per-grid midpoint involved | n/a (correct) |

## 4. The four changes in detail

All edits in `BabenkoRigorous/src/TaylorSeries.jl`.

### 4.1 `sup_bound`

```diff
-    Fbound = ts.sup_bounds[M + 2]
+    Fbound = ts.sup_bounds[M + 2]
+    fbound_radius_arf = Arblib.ubound(Fbound * rem_factor)

     best    = Arb(0; prec=prec)
     best_ub = Arblib.Arf()
     for j in 1:(ts.M_grid)
         s = Arb(0; prec=prec)
         for m in 0:M
             s = s + abs(ts.derivs[j, m+1]) * hpow_over_fact[m+1]
         end
-        s = s + Fbound * rem_factor
+        Arblib.add_error!(s, fbound_radius_arf)
```

### 4.2 `inf_bound`

```diff
-    Fbound = ts.sup_bounds[M + 2]
+    Fbound = ts.sup_bounds[M + 2]
+    fbound_radius_arf = Arblib.ubound(Fbound * rem_factor)

     for j in 1:(ts.M_grid)
         f0 = ts.derivs[j, 1]
         tail = Arb(0; prec=prec)
         for m in 1:M
             tail = tail + abs(ts.derivs[j, m+1]) * hpow_over_fact[m+1]
         end
-        tail = tail + Fbound * rem_factor
-        lb = f0 - tail
+        lb = f0 - tail
+        Arblib.add_error!(lb, fbound_radius_arf)
```

### 4.3 `dmin_bound` — Lipschitz step

```diff
     lip_correction_arb = supnorm_bound(yu_d[2]) + supnorm_bound(Hyu_d[2])
-    d_lower_arb  = Arb(D_lb_min_arf; prec=p) - lip_correction_arb * h
+    d_lower_arb  = Arb(D_lb_min_arf; prec=p)
+    Arblib.add_error!(d_lower_arb, Arblib.ubound(lip_correction_arb * h))
     d_lower_init = Arb(Arblib.lbound(d_lower_arb); prec=p)
```

### 4.4 `dmin_bound` — return value (substantive, not cosmetic)

```diff
     ts_D = sqrt_series(ts_D2, d_lower_init)
-    D_lower = inf_bound(ts_D)
+    ib = inf_bound(ts_D)
+    D_lower = Arblib.lbound(ib) > Arblib.lbound(d_lower_init) ? ib : d_lower_init
     return (D_lower, ts_D)
```

**Why this one is substantive.** `dmin_bound` has two rigorously-valid lower
bounds available — the Lipschitz `d_lower_init` and `inf_bound(ts_D)`. The
old code always returned the latter, which is wildly loose when
`sqrt_series`'s Faà-di-Bruno `sup_bounds` are loose (the high-mode case).
The new code picks the tighter of the two. Both are sound; the choice is
the max-by-`lbound`.

## 5. Faà di Bruno blowup and the `tighten_sup_bounds!` fix

### 5.1 The original problem

`sqrt_series` / `inv_series` originally produced `sup_bounds` for `D^{(m)}`
over-estimated by 10⁹–10¹⁰ at `m = 4–5` for the 110-mode PINN input.

| m | true `‖D^{(m)}‖∞` | original `sup_bounds[m+1]` | ratio |
|---|---:|---:|---:|
| 1 | 5.6 | 322 | 58× |
| 2 | 57 | 4.1 × 10⁵ | 7 × 10³ |
| 3 | 515 | 1.5 × 10⁹ | 3 × 10⁶ |
| 4 | 7 × 10³ | 9.8 × 10¹² | 1.4 × 10⁹ |
| 5 | 1.4 × 10⁶ | 8.7 × 10¹⁶ | 6 × 10¹⁰ |

### 5.2 The `tighten_sup_bounds!` fix (2026-05-22 follow-up)

`tighten_sup_bounds!(ts)` in `TaylorSeries.jl` walks `sup_bounds[m+1]`
down from the loose Faà-di-Bruno top using the grid+Lipschitz recursion
`sup_bounds[m+1] ← max_j |f^(m)(u_j)| + sup_bounds[m+2] · h`.  Wired in as
the final step of `sqrt_series`, `inv_series`, and TaylorSeries `*`.

Stays consistent with the midpoint/radius convention from §1: the new
midpoint is the tight grid max (informative), the radius carries the
Lipschitz contribution (`add_error!` style).

After this fix, low-to-mid `m` ratios go essentially to 1 (e.g.
`ts_y_yu_h` matches truth at every order to within ~1× now).  High-`m`
ratios stay loose because `‖D^(m)‖∞` grows 200–4000× per order at
`m ≥ 5` while the Lipschitz damping is only `h ≈ 3 × 10⁻³`; their
product is `>1`, so per-level damping flips from helpful to harmful at
the top.

### 5.3 N_grid scan with the fix in place

The radius scaling beats the naive `N⁻⁵` because `tighten_sup_bounds!`
uses `h` per level — smaller `h` tightens the sup_bounds chain as well
as the final remainder factor, until the chain itself becomes tight at
high enough N and pure `N⁻⁵` takes over.

At M_order=4, prec=128, extra_orders=0:

| N_grid | midpoint | radius | runtime | effective scaling vs prev N |
|---:|---:|---:|---:|---:|
| 2 048 | 1.58 × 10⁻⁷ | 2.56 × 10¹¹ | 11 s | (baseline) |
| 8 192 | 1.52 × 10⁻⁷ | 6.35 × 10⁵ | 44 s | N⁻⁹·⁹ |
| 32 768 | 1.52 × 10⁻⁷ | 1.96 × 10² | 181 s | N⁻⁸·⁴ |
| 65 536 | 1.52 × 10⁻⁷ | 5.20 | 497 s | N⁻⁵·² (transitioning to N⁻⁵) |

The compound `N⁻⁹` regime dominates while `sqrt_series` sup_bounds are
loose; once they tighten enough, standard `N⁻⁵` from `h^(M+1)` takes
over.  To reach radius ≲ 10⁻⁶ for a useful certificate at this M_order
needs roughly `N ≳ 10⁶`, feasible but expensive (~hours at prec=128).

### 5.4 `tight_DG_sup_bounds` — algebraic G/D recurrence (2026-05-22 follow-up)

Implemented as a parallel sup-bound source for `Gpp_bound_Gu` /
`Gppp_bound_Gu`, computed via the algebraic recurrences from
`D · D = A² + B²` and `G · D = B`:

```
‖D^(m)‖∞ ≤ [‖(D²)^(m)‖∞ + Σ_{k=1..m-1} C(m,k)·D_sup[k+1]·D_sup[m-k+1]] / (2·D_lower)
‖G^(m)‖∞ ≤ [‖B^(m)‖∞   + Σ_{k=0..m-1} C(m,k)·G_sup[k+1]·D_sup[m-k+1]] / D_lower
```

where the trig-poly sup_bounds `‖(D²)^(m)‖∞`, `‖B^(m)‖∞` come from exact
TrigPoly multiplication + iterated `deriv` + `supnorm_bound` (triangle
inequality on the coefficient vector — exact for trig polys). Requires
adding `Base.:*(::TrigPoly, ::TrigPoly)` to `FourierSeries.jl`, also
implemented.

Compared to `sqrt_series`/`inv_series`'s Faà-di-Bruno chain (with
`tighten_sup_bounds!` enabled), the algebraic recurrence is:

| order | Faà di Bruno + tighten | `tight_DG_sup_bounds` |
|---:|---:|---:|
| D^(5) | ratio 1.1 × 10⁷ over truth | **ratio 4.4 × 10⁶** |
| D^(8) | ratio 2.9 × 10⁶ | **ratio 6.0 × 10⁵** |
| G^(7) | ratio comparable | comparable |
| G^(8) | **ratio 2.6 × 10⁹ (chain better)** | ratio 6.5 × 10⁹ |

`tight_DG_sup_bounds` strictly beats Faà di Bruno at low/mid `m`
(orders of magnitude better) but is comparable or slightly worse at the
specific high-`m` entries the chain consumes (`G^(7)`, `G^(8)`), because
the algebraic recurrence's quadratic accumulator term `Σ G·D` amplifies
loose intermediate bounds.

`_residual_bound_full` now uses `min(chain, algebraic)` for
`Gpp_bound_Gu` / `Gppp_bound_Gu`, picking whichever is tighter per
quantity. The choice is informationally non-trivial but materially small
at the consumed orders.

### 5.5 Where the radius stands

After all fixes (convention restoration, `tighten_sup_bounds!`,
`tight_DG_sup_bounds`, `min`-selection) at M_order=4, prec=128,
extra_orders=0:

| N_grid | radius | scaling note |
|---:|---:|---|
| 2 048 | 1.81 × 10¹¹ | (mostly from Gppp_bound_Gu = 9.8 × 10²⁵ via chain) |
| 8 192 | 6.35 × 10⁵ | Gppp shrinks too (N-dependent through `h` in `tighten`) |
| 32 768 | 1.96 × 10² | combined ≈ N⁻⁷ scaling |
| 65 536 | 5.20 | (N⁻⁵ asymptote taking over) |

To reach a **useful** rigorous bound (radius below the midpoint 10⁻⁷)
needs roughly `N ≈ 500k`, ~3 hours at prec=128. Still feasible. The
midpoint already perfectly tracks the FFT truth at N=2048; what's left
is brute force on the radius.

The remaining looseness is now structural: the consumed `Gppp_bound_Gu`
is `‖G^(8)‖∞ ≈ 10¹⁶` vs the bound ≈ 10²⁵–10²⁶, i.e. 10⁹–10¹⁰× loose. No
algorithm short of explicit FFT-based estimation of `G^(8)`'s sup-norm
on a fine grid (which is what brute-force N effectively does) can do
better. Direct algebraic + grid+Lipschitz both bottom out around this
level because of the quadratic compounding in the recurrence on a
function whose 110-mode content produces `‖G^(8)‖∞ ≈ k^8 max|G|` ≈ 10¹⁶
with the worst-case input bound only 10²–10³× away from there.

None of these affect the Lean side: the `sup_bounds` semantics is
"rigorous upper bound on the derivative", which any tighter algorithm
satisfies.

### 5.6 Threading (2026-05-22 follow-up)

`Threads.@threads` added to the per-grid loops in `trigpoly_to_taylor`,
`Base.:*(::TaylorSeries, ::TaylorSeries)`, `inv_series`, `sqrt_series`,
`dmin_bound`, and the `_residual_bound_full` step-6b loop. Follows the
cap-ccf `hermite_functions.jl` pattern of `Threads.@threads for pc in
0:(n_pc-1)` over disjoint point chunks with no shared mutable state.

The one loop that needed restructuring is `dmin_bound`'s min-tracking
loop: replaced the in-place `if Dj_lb < D_lb_min_arf` update with a
per-grid `Dj_lbs[j] = ...` write and a serial reduction afterward (same
pattern cap-ccf uses for thread-safe reductions).

Run Julia with `--threads=10` to use 10 cores. Speedup is roughly 2–3×
at N=2048 (much serial overhead from sup_bound computation and
single-threaded `tighten_sup_bounds!` recursion), growing to 5–8× at
large N where the per-grid loops dominate. Below, the threading
slowdown estimate at small N is mostly the cost of paying for
Faà-di-Bruno's serial sup_bounds construction (independent of N).

### 5.7 N_grid scan, threaded (10 threads)

TBD pending the running sweep — adding here once N=131072+ results are in.

`tight_DG_sup_bounds`'s `min`-selection has negligible effect at the
consumed entries — the chain (with `tighten_sup_bounds!`) wins there.
It strictly tightens low-to-mid `m` entries, but those aren't what
flows into the final radius.

## 6. Implications for the Lean side — what I actually checked

I read the headline statement (and most of the proof) of every Lean file
named in `WaveCAP/README.md`'s status table:
`TaylorRemainder.lean`, `GridToSupNormTaylor.lean`, `DLowerBound.lean`,
`FaaDiBrunoBound.lean`, `AliasingErrorBound.lean`, `Definitions.lean`,
`Certificate.lean`, `Main.lean`. I also grepped the entire
`WaveCAP/WaveCAP/` tree for `add_error`, `midpoint`, `radius`, `Arb` —
zero hits.

**Findings:**

- All eight files state their lemmas at the **real-number level** using
  Mathlib's `iteratedDeriv`, `abs`, `≤`, `Finset.sum`, `Finset.prod`. No
  theorem encodes a specific Arb decomposition.
- In particular, `grid_to_supNorm_taylor` takes its `M_bound` hypothesis
  as `∀ j, (Σ |f^(k)(u_j)|·h^k/k!) + F·h^(M+1)/(M+1)! ≤ M_bound` — which
  is exactly the formula the fixed Julia `sup_bound` realizes via its
  per-`j` Arb (`midpoint = Σ taylor terms`, `radius = Fbound · rem_factor`).
- `Certificate.lean` exports `ε₀_cert, DLower_cert : ℝ` and the trust
  anchors `cert_residual_le`, `cert_D_ge` — pure real-number statements;
  whichever Julia algorithm produces `ε₀_cert` and `DLower_cert` is fine,
  as long as the rigorous bound semantics holds.

**Therefore, no Lean changes are required by today's Julia fix.**

**Gaps I did notice while reading (not bugs from today's change, but
worth flagging for the Lean roadmap):**

1. **No `grid_to_infNorm_taylor` exists.** There's a sup-bound grid wrapper
   but no inf-bound counterpart. `dmin_bound`'s inf-bound construction
   is currently trusted by Lean only obliquely through the
   `cert_D_ge : ∀ u, DLower_cert ≤ D u` axiom. If we want to remove that
   axiom in favor of a derived theorem, an `inf_bound` Lean lemma needs
   writing.
2. **`R` in `Definitions.lean` is parameter-agnostic.** It takes abstract
   real `μ, β`. The PINN→Babenko conversion (`μ_cert = lamb²`,
   `β_cert = beta²`) is the Julia certificate-emitter's job — the Lean
   side just takes whatever values it gets. When `Certificate.jl`'s
   emitter is wired up for the PINN run, **make sure it writes
   `μ_cert = lamb^2` and `β_cert = beta^2`, not the raw PINN values.**
3. **No reflective Arb tactics in Lean.** Confirmed by grep — there is no
   `decide`-based or `norm_num`-extension proof that mirrors the Julia Arb
   construction step-for-step, so the `f0 − tail` → `f0 + add_error(tail)`
   reshape has no Lean-side counterpart that would need updating.

**What I did not exhaustively check:**

- I did not read every step of every proof. I read the statements and
  the proof outlines. The README claims all files compile sorry-free
  against Mathlib `v4.29.0-rc6` aside from the documented N3/N4
  packaging axiom — I took that on faith rather than rebuilding.
- I skimmed `FourierTailHilbert.lean` (74 KB) and the Hilbert files
  (`HilbertOnTrigPoly.lean`, `HilbertCommutesDeriv.lean`) at the level
  of headings only.
- `Common.lean` and `Foundations.lean` are short stubs — only glanced at.

If the Lean side has correctness issues, they would be unrelated to the
midpoint/radius convention this doc is about, since none of the
real-number statements I saw depend on the Arb representation.

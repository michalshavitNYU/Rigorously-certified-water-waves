# Milestones and Validation Tests

Rigorous residual bound for the Babenko equation — development roadmap.

> **Historical document.** This roadmap was superseded by the as-built
> design: the Julia pipeline (M1–M7) and the Lean side landed 2026-05-22/24
> with a different endgame than M12/M13 envisioned — the Phase-6
> trust-anchor design, completed 2026-07-03, leaves exactly **two** axioms
> (the Arb-computed scalars `DLower_cert`, `ε₀_cert`), with all structural
> steps machine-checked. See `status_2026-05-24.md` and
> `audit_2026-07-03.md` for the current state; the milestone statuses
> below were never updated and understate what is done.

Each milestone has a clear deliverable, concrete runnable tests, and explicit exit criteria. Tests are grouped as *algebraic* (identity checks that should hold exactly), *numerical* (comparison against Float64 baseline or analytic reference), and *certification* (Arb ball contains the expected value with a tight radius).

Milestones M1–M7 are Julia; M8–M12 are Lean; M13 is integration. M0 is already complete.

---

## Standard cross-check: random coefficients + Float64 containment

The actual PINN approximate solution is not yet available. Until it is, all Julia milestones
use **random finite cosine series** as inputs. This is sufficient because correctness of the
interval arithmetic does not depend on the input being a good approximate solution — any
finite cosine series is a valid input and will produce some residual.

The universal validation invariant is:

> **The certified Arb upper bound on ‖R‖∞ must be ≥ the Float64 estimate from `babenko_residual.jl`.**

This is a necessary condition for correctness (the Arb bound must contain the true value, which
the Float64 approximation estimates). It can be checked mechanically for any input without
knowing what the "right" answer is.

A standard random-coefficient test harness (used across M1–M6):

```julia
function random_test(N=8, μ=1.3, β=0.05; seed=42, prec=128, N_grid=256, M_order=4)
    rng = MersenneTwister(seed)
    # Random coefficients decaying as 1/k (mimics a smooth function)
    a_coeffs = [randn(rng) / k for k in 1:N]
    
    # Float64 reference residual (from existing babenko_residual.jl)
    x = range(0, 2π; length=512+1)[1:512]
    y_f64 = sum(a_coeffs[k] * cos(k*x) for k in 1:N)
    R_f64_sup = maximum(abs.(babenko_residual_f64(y_f64, μ, β)))
    
    # Arb certified bound
    R_arb_bound = residual_bound(a_coeffs, μ, β; prec, N_grid, M_order)
    
    # The invariant
    @test Arblib.ubound(R_arb_bound) >= R_f64_sup  # Arb bound contains Float64 estimate
    @test Arblib.isfinite(R_arb_bound)              # Not ±∞
    @test Arblib.ispositive(R_arb_bound)            # Positive
    
    return R_f64_sup, R_arb_bound
end
```

Run `random_test` with 10–20 different seeds at each milestone to build confidence. The ratio
`Arb_upper / Float64_estimate` is a useful diagnostic: it should be > 1 (correctness) and
ideally < 10 (not too pessimistic). A ratio > 100 signals that some error term is being
over-estimated and should be investigated.

---

## Dependency graph

```
M0 (baseline)
│
├── M1 (FourierSeries)
│         │
│   M2 (TaylorSeries  ── includes inv_series/sqrt_series/dmin_bound)
│         │
│         ├── M3 (GDerivatives) ─────── M4 (FFTHilbert)
│         │             │                       │
│         │             └──────────► M5 (Residual β=0)
│         │                              │       (uses Path B for H[y·y_u])
│         │                              ▼
│         └─────────────────────────► M6 (Residual β≠0)
│                                        │
│                                       M7 (Certificate)
│                                        │
M8  (HilbertOnTrigPoly + CommutesDeriv) ── M12 (Integration)
M9  (AliasingErrorBound: integral-test) ──┤
M10 (FaaDiBrunoBound — new)              │
M11 (TaylorRemainder — order-M form)     │
                                          ▼
                              M11 (Main.lean skeleton)

```

**Note (post-refactor):** M5 (β=0 residual) now depends on M3 (GDerivatives) and
M4 (FFTHilbert) — *not* only M2 — because `H[y·y_u]` is routed through
`fft_hilbert_series` (Path B) for engineering uniformity with the capillary path.
The dependency arrows above reflect this.  Two new Lean milestones are introduced
by the audit: M10 (`FaaDiBrunoBound.lean`, the Bell-polynomial bound used by
`inv_series` and `sqrt_series`) and a revised M9 (`AliasingErrorBound.lean` with
the integral-test form `4·Cp/((p-1)·(N/2-1)^{p-1})` plus a Fourier-tail π²/3
lemma; replaces the original `ζ(p)/N^p` form).

M7–M10 are pure Lean analysis, independent of all Julia milestones. They can be
developed in parallel with M1–M6. M12 assembles M8–M11. M13 connects M7 + M12.

---

## M0 — Baseline (complete)

**Deliverable:** Float64 prototype: `hilbert.jl`, `babenko_residual.jl`, `hilbert_plots.jl`.

**Passing tests:**
- `julia hilbert.jl`: H[cos(nx)] = sin(nx) to machine precision for n = 1,3,5,10.
- `julia babenko_residual.jl`: three residual cases (random, Stokes β=0 ε=0.15, grav-cap β=0.05 ε=0.05) all print without error, residual norms match expected order-of-magnitude.

These Float64 values become reference targets for M1–M6.

---

## M1 — `FourierSeries.jl`

**Deliverable:** `TrigPoly` struct with exact Arb-ball arithmetic: `hilbert`, `deriv`, `*`, `eval`, `supnorm_bound`.

### Tests

**T1.1 — Algebraic: H² = −I**
For a random degree-8 trig poly `f` with Float64-seeded Arb coefficients, verify `hilbert(hilbert(f)) + f` has all Arb coefficients whose balls contain zero.

```julia
f = random_trigpoly(8, prec=128)
g = hilbert(hilbert(f))
@test all(Arblib.contains_zero(g.a[k] + f.a[k]) for k in eachindex(f.a))
@test all(Arblib.contains_zero(g.b[k] + f.b[k]) for k in eachindex(f.b))
```

**T1.2 — Algebraic: H and ∂ commute on trig polys**
`hilbert(deriv(f))` and `deriv(hilbert(f))` should give the same trig poly.

```julia
@test trigpoly_equal(hilbert(deriv(f)), deriv(hilbert(f)))
```

**T1.3 — Algebraic: product rule**
`deriv(f * g) == deriv(f)*g + f*deriv(g)` for random degree-4 polys f, g.

**T1.4 — Bernstein bound is genuine**
Evaluate `f` at 1000 equally-spaced Arb points. Check every |f(u_j)| ≤ supnorm_bound(f).

**T1.5 — Matches Float64 baseline**
For `y = 0.3cos(u) + 0.15cos(2u) + 0.07cos(3u)` (same as M0's random case), evaluate
`hilbert(deriv(y_tp))` at 100 grid points using `TrigPoly`, compare to `periodic_hilbert` from
`hilbert.jl`. Each Arb ball should contain the Float64 value.

**Exit criteria:** all five tests pass; no `Arblib.isnan` or infinite radius balls produced.

---

## M2 — `TaylorSeries.jl`

**Deliverable:** `TaylorSeries` struct (ported from cap-ccf) carrying a full
`sup_bounds :: Vector{Arb}` of length `M_order+2`, with:
- arithmetic: `*` (exact Leibniz on sup_bounds vectors), `+`, scalar `*`, `-`
- `deriv_series` (shift sup_bounds left by one — no inflation)
- `truncate_order` (restrict to lower order)
- `inv_series` and `sqrt_series` (Faà di Bruno via `_faa_di_bruno_derivs` =
  ArbSeries composition with analytic supnorms of `1/x` and `√x`)
- `dmin_bound` (certified lower bound on `D` via sqrt_series; throws on non-positive)
- `trigpoly_to_taylor` (TrigPoly → TaylorSeries with per-derivative `supnorm_bound`)
- `sup_bound` / `inf_bound` (Taylor-remainder form, max over grid + `sup_bounds[end]·h^{M+1}/(M+1)!`)

### Tests

**T2.1 — Taylor evaluation contains true value**
Build a `TaylorSeries` for `sin(u)` at 32 grid points (compute coefficients analytically).
For 200 random test points `u` (not grid points), check `evaluate_on_subinterval(ts, u)` contains `sin(u)`.

**T2.2 — Derivative shift**
`derivative(ts_sin)` should agree with `ts_cos` (TaylorSeries for cos) at all grid points to within Arb precision.

**T2.3 — Product rule via TaylorSeries multiplication**
Build Taylor series for `sin(u)` and `cos(u)`. Their product should agree with Taylor series of `sin(u)cos(u) = sin(2u)/2`. Check all derivative-level values match.

**T2.4 — Sup bound is genuine**
For `ts_sin`, evaluate `sin` at 10 000 points. Check `max |sin(u_j)| ≤ ts_sin.sup_bounds[1]`.

**T2.5 — Taylor remainder decreases with order**
For the same function, build Taylor series at order M = 2, 4, 6. The sup_bound on the remainder
term should decrease roughly as `h^{M+1}/(M+1)!` where `h = 2π/N_grid`.

**Exit criteria:** T2.1 passes for all 200 test points; T2.5 shows expected convergence rate.

---

## M3 — `GDerivatives.jl`

**Deliverable:** `compute_G_grid(y::TrigPoly, dmin::DMinResult, M_order::Int, u_grid) -> GGrid`
returning `{G^{(m)}(u_j)}` for m = 0,...,M_order.

### Tests

**T3.1 — G^{(0)} matches direct evaluation**
For `y = 0.3cos(u) + 0.15cos(2u)`, compute `G^{(0)}(u_j) = (1 + H[y_u](u_j)) / D(u_j)` two ways:
(a) from `GDerivatives.jl`; (b) directly using `TrigPoly.eval`. Check agreement to within Arb radii.

**T3.2 — G^{(1)} validated by finite differences**
Let `h = 1e-5`. Check `|G^{(1)}(u_j) − (G^{(0)}(u_j+h) − G^{(0)}(u_j−h))/(2h)| = O(h²)` at
each grid point (using Float64 as reference, Arb as rigorous). Expected error ~`h² · ‖G^{(2)}‖/6`.

**T3.3 — G^{(2)} validated by finite differences**
Same as T3.2 for the second derivative.

**T3.4 — Analytical case**
For `y = ε·cos(u)`:
- Numerator = `1 + ε·cos(u)`, D = as in M2.
- `G(u) = (1 + ε·cos(u)) / √(ε²sin²u + (1+εcosu)²)`.
- Compute `G^{(0)}` at u = 0, π/2, π analytically. Check Arb balls contain these values.

**Exit criteria:** T3.1 absolute error < 1e-12 (Arb precision); T3.2/T3.3 FD error proportional to h².

---

## M4 — `FFTHilbert.jl`

**Deliverable:** `fft_hilbert(G_vals::Vector{Arb}, N_fft::Int, p::Int, Gp_sup::Arb) -> Vector{Arb}`
returning `{H[G](u_j)}` with aliasing error already added to each ball's radius.

Also the TaylorSeries version: given `GGrid`, returns `TaylorSeries` for `H[G]`.

### Tests

**T4.1 — Trig poly input: near-zero aliasing**
For `G(u) = sin(3u)` (a degree-3 trig poly), `H[G] = -cos(3u)` exactly. Run `fft_hilbert` with
`N_fft = 256`. Check: (a) every output ball contains the exact value `-cos(3u_j)`; (b) the aliasing
error estimate is < 1e-10 (the trig poly Fourier tail is zero, so aliasing = 0 up to Arb rounding).

**T4.2 — Non-poly function: convergence in N_fft**
For `G(u) = (1 + cos(u)) / (2 + sin(u))`, compute `H[G]` at Float64 via large-N (N=65536) DFT as
reference. Run `fft_hilbert` at N_fft = 256, 512, 1024, 2048. Check:
(a) Each Arb ball contains the Float64 reference. (b) The Arb ball radius decreases as `N_fft^{−p}`.

**T4.3 — Aliasing bound is rigorous**
From T4.2 at N_fft=256: the aliasing bound `ε_alias` printed by `fft_hilbert` should satisfy
`|H_N[G](u_j) − H[G](u_j)| ≤ ε_alias` for all j (verified against the Float64 large-N reference).

**T4.4 — TaylorSeries output: derivatives match finite differences**
Build `TaylorSeries` for `H[G]` via `fft_hilbert` and check that the stored derivatives at grid
points match finite differences of `H[G]` at Float64 precision, level by level.

**T4.5 — Error budget: aliasing dominates Arb rounding**
At N_fft = 2048, print the ratio `aliasing_error / arb_rounding_error` for a typical input.
Aliasing should dominate (ratio >> 1), confirming the DFT itself is not the bottleneck.

**T4.6 — Containment on random G inputs**
For 10 random smooth periodic functions G (e.g., `G(u) = Σ_{k=1}^{8} r_k·cos(ku)/(k²)` for
random `r_k`), compute `H[G]` via FFTHilbert (Arb) and via the Float64 `periodic_hilbert` from
`hilbert.jl`. Check every Arb output ball contains the Float64 value at every grid point.

**Exit criteria:** T4.1 all balls contain exact value; T4.2 N^{−p} convergence confirmed; T4.3 bound is rigorous; T4.6 all balls contain Float64 reference.

---

## M5 — `ResidualBound.jl`, β = 0

**Deliverable:** `residual_bound(y_coeffs, μ, β=0; M_order, N_grid, prec) -> Arb` returning a
certified upper bound on `‖R‖_∞`. The capillary terms are skipped when β = 0.

### Tests

**T5.1 — Small-amplitude Stokes wave (primary regression test)**
`y = −ε²/2 + ε·cos(u) + ε²·cos(2u)`, `μ = 1 + ε²`, `β = 0`, `ε = 0.15`.
Float64 baseline: `‖R‖_∞ ≈ 1.14e-2`.
- Check: certified bound is a positive Arb value.
- Check: certified bound ≤ 0.05 (rough sanity; should be ~3× the Float64 value after Taylor remainder).
- Check: certified bound is finite and has tight radius (radius / midpoint < 0.1).

**T5.2 — Bound contains Float64 estimate**
The certified upper bound must be ≥ the Float64 estimate `1.14e-2`. (Arb bound is an *upper* bound.)

**T5.3 — Convergence in ε**
Run for ε = 0.01, 0.05, 0.10, 0.15. Plot certified bound vs ε. Expect O(ε³) growth (the next
Stokes term). Confirm slope ≈ 3 on log-log.

**T5.4 — Convergence in Taylor order**
Fix ε = 0.15, vary M_order = 2, 4, 6, 8. Certified bound should decrease (more terms → tighter
remainder), then plateau (residual itself is the floor).

**T5.5 — β=0 correctness: capillary contribution is zero**
With β = 0, the capillary contribution (`β·∂_u(y_u/D)` and `β·H[∂_u G]`) vanishes and the
code takes the gravity-only branch. Note: `fft_hilbert_series` *is* still invoked for
`H[y·y_u]` (Path B; see Path A/B discussion in the overview), so this test verifies only
that `GDerivatives` and the capillary block are skipped — not that `FFTHilbert` is entirely
absent.

**T5.6 — Random coefficient containment (10 seeds)**
Run `random_test(N=8, β=0, seed=s)` for s = 1,...,10. All 10 must satisfy
`Arb_upper ≥ Float64_estimate`. Print the ratio for each; flag any ratio > 50.

**Exit criteria:** T5.1 passes; T5.3 shows ε³ growth; T5.4 shows decreasing-then-plateauing bound; T5.6 all 10 seeds pass.

---

## M6 — `ResidualBound.jl`, β ≠ 0

**Deliverable:** Full residual bound including capillary terms (Path B enters).

### Tests

**T6.1 — Linear gravity-capillary mode (primary regression test)**
`y = ε·cos(u)`, `μ = β + 1`, `β = 0.05`, `ε = 0.05`.
Float64 baseline: `‖R‖_∞ ≈ 3.69e-3`.
- Check: certified bound is positive and finite.
- Check: certified bound ≤ 0.02.
- Check: bound contains 3.69e-3.

**T6.2 — Setting β=0 recovers M5 result**
With same y, μ as M5 T5.1 but β=0.0 passed to M6's code, the result should match M5's output.

**T6.3 — D positivity is certified**
Before computing capillary terms, `DMinBound` must succeed with a positive lower bound. Print
`DLower` and confirm > 0.01.

**T6.4 — Aliasing error is subdominant**
Print the aliasing error contribution and the Path A contribution to the final bound separately.
Aliasing should be < 10% of total bound at N_fft = 2048 (confirming N_fft is adequate).

**T6.5 — Convergence in N_fft**
Fix all other parameters; vary N_fft = 512, 1024, 2048, 4096. Certified bound should decrease
until it plateaus (when aliasing < Path A terms). Plot to confirm.

**T6.6 — Random coefficient containment, β≠0 (10 seeds)**
Run `random_test(N=8, β=0.05, seed=s)` for s = 1,...,10. All 10 must satisfy
`Arb_upper ≥ Float64_estimate`. This is the key test that the full capillary
path (GDerivatives + FFTHilbert) is correctly assembled. Print ratio for each seed.

**Exit criteria:** T6.1 passes; T6.2 matches M5 exactly; T6.3 positive; T6.6 all 10 seeds pass.

---

## M7 — `Certificate.jl`

**Deliverable:** `write_certificate(path, y_coeffs, μ, β, result::ResidualResult)` writing
`Certificate.lean` suitable for import into Lean.

### Tests

**T7.1 — Lean parses the output**
Create a minimal `lakefile.toml` importing only `Certificate.lean`. Run `lake build`. Expect zero errors.

**T7.2 — Values round-trip**
Parse the decimal literal strings back to Float64. Check they agree with the Julia Arb midpoints
to at least 15 significant digits.

**T7.3 — `is_certified` flag logic**
Run with a deliberately-too-small N_grid (so Taylor remainder is large). Check `is_certified = false`
and the output file has a comment warning.

**T7.4 — Determinism**
Run the certificate writer twice with identical inputs. Output files should be byte-for-byte identical
(no timestamps in the numeric content).

**Exit criteria:** T7.1 compiles; T7.2 passes for all constants.

---

## M8 — `GridToSupNorm.lean`

**Deliverable:** Lean 4 theorem (no `sorry`):
```lean
theorem lipschitz_sup_norm_of_grid_bound {f : ℝ → ℝ} {T L ε₀ : ℝ} {M : ℕ}
    (hM  : 0 < M)
    (hL  : LipschitzWith (Real.toNNReal L) f)
    (hgrd : ∀ j : Fin M, |f (T * j / M)| ≤ ε₀)
    : ∀ u ∈ Set.Icc 0 T, |f u| ≤ ε₀ + L * T / (2 * M)
```

### Tests

**T8.1 — Compiles with zero `sorry`s**
`lake build BabenkoWave.GridToSupNorm` prints no errors and `#check lipschitz_sup_norm_of_grid_bound`.

**T8.2 — Concrete instantiation**
Prove in a `#check` block: for `f = sin`, `T = 2π`, `L = 1`, `M = 10`, `ε₀ = sin(π/10)`,
the bound gives `|sin u| ≤ sin(π/10) + π/10` — a provable (if weak) statement.

**T8.3 — Boundary case: M = 1**
Instantiate with M = 1 (one grid point). The bound degenerates to `ε₀ + L·T/2`, which is still correct.

**Exit criteria:** T8.1 zero `sorry`s; T8.2 example compiles.

---

## M9 — `HilbertOnTrigPoly.lean` + `HilbertCommutesDeriv.lean`

**Deliverable:**
- `H_of_cosine_series` definition and `hilbert_cos_eq_sin` proved for each mode k.
- `hilbert_commutes_deriv` proved for trig polynomials.

### Tests

**T9.1 — Mode-by-mode identities compile**
```lean
example : H_of_cosine_series (fun _ => (1 : ℝ)) (fun j => if j = 0 then 1 else 0) u
          = Real.sin u := by simp [H_of_cosine_series]
```

**T9.2 — Linearity: H[f + g] = H[f] + H[g]**
Proved as a lemma, no `sorry`.

**T9.3 — Commutativity for degree-2 example**
For `f(u) = a·cos(u) + b·cos(2u)`, prove `H[∂f] = ∂H[f]` by explicit computation (both sides
equal `a·cos(u) + 2b·cos(2u)` after simplification).

**Exit criteria:** T9.1–T9.3 all compile with zero `sorry`s.

---

## M10 — `AliasingErrorBound.lean`

**Deliverable:** Lean theorem bounding the DFT aliasing error for a smooth periodic function with
bounded p-th derivative:
```lean
theorem aliasing_error_bound {f : ℝ → ℝ} {p N : ℕ} {C : ℝ}
    (hf  : ContDiff ℝ p f)
    (hper : Function.Periodic f (2 * π))
    (hC  : ∀ u, |iteratedDeriv p f u| ≤ C)
    (k   : ℤ)
    : |dft_coeff f N k - fourierCoeff f k| ≤ C * 2 * riemannZeta p / (2 * π * N ^ (p - 1))
```

(Exact statement may differ based on Mathlib's Fourier coefficient convention; the key inequality
is that the aliasing error decays as N^{−(p−1)}.)

### Tests

**T10.1 — Compiles**
`lake build BabenkoWave.AliasingErrorBound` with at most well-documented `sorry` markers for any
Poisson summation gap (expected: 1 `sorry` at most for the summation formula itself).

**T10.2 — Corollary instantiation**
Prove a corollary: for `p = 4`, `N = 1024`, `C = 1`, the aliasing error per frequency is ≤
`2·ζ(4)/(2π·1024³)`. This is a purely arithmetic check that Lean can verify by `norm_num`.

**Exit criteria:** T10.1 with ≤ 1 documented `sorry`.

---

## M11 — `TaylorRemainder.lean`

**Deliverable:** Taylor remainder lemma (adapted from CCF's `TaylorRemainderSubinterval.lean`):
```lean
theorem taylor_remainder_sup_norm {f : ℝ → ℝ} {M : ℕ} {a b F : ℝ}
    (hf  : ContDiffOn ℝ (M + 1) f (Set.Icc a b))
    (hF  : ∀ x ∈ Set.Icc a b, |iteratedDerivWithin (M + 1) f (Set.Icc a b) x| ≤ F)
    : ∀ x ∈ Set.Icc a b,
        |f x - Σ_{k=0}^{M} iteratedDerivWithin k f (Set.Icc a b) a / k! * (x - a)^k|
        ≤ F * (b - a)^(M + 1) / (M + 1)!
```

### Tests

**T11.1 — Compiles, no sorry (or matches CCF exactly)**
If the CCF version (`/Users/buckmaster/Lean/ccf-lean-poc`) has a proved version, import and restate it.
The test is that the import compiles and the restated theorem type-checks.

**T11.2 — Concrete instantiation for sin**
For `f = sin`, `M = 3`, `[a,b] = [0, π/4]`, `F = 1`, the bound gives `|sin(x) − (x − x³/6)| ≤ (π/4)⁴/24`.
Verify this is provable by `norm_num` after instantiation.

**Exit criteria:** T11.1 compiles; T11.2 `norm_num` closes the goal.

---

## M12 — `Main.lean` skeleton

**Deliverable:** Full proof structure with `axiom` placeholders for the three Julia hypotheses.
The theorem compiles to completion using the axioms (no `sorry`, axioms are not `sorry`s).

```lean
-- Axioms (to be replaced by Certificate.lean in M13)
axiom julia_grid_cert  : ∀ j : Fin Certificate.M, |R (2 * π * ↑j / Certificate.M)| ≤ Certificate.ε₀
axiom julia_deriv_cert : ∀ u : ℝ, |deriv R u| ≤ Certificate.L
axiom julia_D_cert     : ∀ u : ℝ, Certificate.DLower ≤ D u

theorem babenko_residual_bound : ∀ u : ℝ, |R u| ≤ Certificate.ε₀ + Certificate.L * π / Certificate.M :=
  ...  -- uses GridToSupNorm, TaylorRemainder, AliasingErrorBound, etc.
```

### Tests

**T12.1 — Compiles with axioms**
`lake build BabenkoWave` with the three axioms in scope. Zero errors, zero `sorry`s.

**T12.2 — Axiom count**
`grep -r "axiom" BabenkoWave/` returns exactly 3 lines (the three Julia certificate axioms).
No hidden `sorry` anywhere.

**T12.3 — All M8–M11 lemmas are actually used**
`grep -r "lipschitz_sup_norm_of_grid_bound\|aliasing_error_bound\|taylor_remainder\|hilbert_commutes"
BabenkoWave/Main.lean` returns at least one hit each. (No dead lemmas.)

**Exit criteria:** T12.1 passes; T12.2 exactly 3 axioms; T12.3 all lemmas used.

---

## M13 — Full integration

**Deliverable:** Replace the three axioms with values from `Certificate.lean`, generated by running
the Julia package on the small-amplitude Stokes wave test case. The Lean proof is fully closed.

### Tests

**T13.1 — End-to-end Julia run**
```
julia scripts/run_stokes.jl
```
Produces `Certificate.lean` in < 60 seconds at prec=128, N_grid=512, M_order=4. The file prints
`is_certified = true` and `residual_bound` is a positive Arb value.

**T13.2 — Lean proof closes**
Import the generated `Certificate.lean`. Delete the three `axiom` lines from M12.
`lake build BabenkoWave` compiles with zero errors, zero `sorry`s, zero axioms.

**T13.3 — Theorem statement is closed and self-contained**
`#check @babenko_residual_bound` prints a theorem with no free hypotheses. The bound `ε` is
a concrete real-number literal from `Certificate.lean`.

**T13.4 — Reproducibility**
Delete `Certificate.lean`, re-run the Julia script, rebuild Lean. The proof still compiles and
the bound matches the previous run to 15 significant digits.

**Exit criteria:** T13.2 zero errors, zero axioms; T13.3 self-contained theorem.

---

## Summary table

| Milestone | Component | Key deliverable | Blocking test | Est. difficulty |
|---|---|---|---|---|
| M0 | — | Float64 prototype | Existing output | complete |
| M1 | Julia | `FourierSeries.jl` | T1.1 H²=−I | low |
| M2 | Julia | `TaylorSeries.jl` | T2.1 evaluation contains true value | low |
| M3 | Julia | `GDerivatives.jl` | T3.2 FD validation of G' | medium |
| M4 | Julia | `FFTHilbert.jl` | T4.2 N^{−p} convergence | medium |
| M5 | Julia | Residual β=0 | T5.1 Stokes wave bound | medium |
| M6 | Julia | Residual β≠0 | T6.1 grav-cap bound | medium |
| M7 | Julia | `Certificate.jl` | T7.1 Lean parses output | low |
| M8 | Lean | `GridToSupNorm` | T8.1 zero sorries | medium |
| M9 | Lean | Hilbert on trig polys | T9.1 mode identities | low |
| M10 | Lean | `AliasingErrorBound` | T10.1 ≤1 sorry | hard |
| M11 | Lean | `TaylorRemainder` | T11.1 adapts from CCF | low |
| M12 | Lean | `Main.lean` skeleton | T12.2 exactly 3 axioms | medium |
| M13 | Both | Full integration | T13.2 zero axioms | medium |

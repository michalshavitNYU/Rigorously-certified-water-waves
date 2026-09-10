# Babenko Water-Wave CAP — Project Overview

**Last updated:** 2026-05-21 (§§6–7 describe the *original plan*; the
as-built Lean design differs — namespace `WaveCAP`, direct
`≤ ε₀_cert` theorem with exact rational literals, Phase-6 two-axiom
trust base. For current status see `status_2026-05-24.md` and
`audit_2026-07-03.md`.)
**Detailed specs:** `lean_design.md`, `julia_design.md` (in this folder)

---

## 1. Project Goal

We seek a computer-assisted proof (CAP) of the following statement:

> Let `y(u) = Σ_{k=1}^{N} a_k cos(ku)` be a fixed finite even cosine series and let `μ, β > 0` be given
> rational parameters. Define the residual of the Babenko equation with surface tension by
>
> ```
> R(u) = μ H[y_u] − y − y H[y_u] − H(y y_u) + β ∂_u(y_u/D) − β H[∂_u((1+H[y_u])/D)]
> ```
>
> where `D = √(y_u² + (1+H[y_u])²)` and `H` is the `2π`-periodic Hilbert transform.
> Then `‖R‖_{L∞([0,2π])} ≤ ε` for an explicit certified constant `ε`.

The proof is split between Julia (rigorous arithmetic via interval ball arithmetic) and Lean 4
(logical scaffolding that imports the certificate and deduces the bound). Neither component is
trusted alone: Julia performs all the hard computation with certified error enclosures; Lean
checks the logic that turns those enclosures into a theorem.

**Scope.** The deliverable is the residual sup-norm bound above — nothing more. The bound is
the quantitative *input* one would need for a downstream Newton–Kantorovich / contraction
argument to upgrade "small residual" into "existence of an exact gravity-capillary Stokes wave
nearby" — but inverting the linearisation of the Babenko operator (an unbounded
infinite-dimensional pseudodifferential operator) is a separate, paper-level problem and is
**out of scope** here. See §10 for what such a follow-up would entail.

---

## 2. Background

**Stokes waves** are steady periodic surface water waves traveling at constant speed. Finding
them amounts to solving a nonlinear integro-differential equation for the free surface profile.
Babenko (1987) reformulated this as a single scalar equation on the conformal parameter `u`:

```
μ H[y_u] = y + y H[y_u] + H(y y_u) − β ∂_u(y_u/D) + β H[∂_u((1+H[y_u])/D)]
```

Here `μ` is the (rescaled) wave speed, `β = T/(gλ²)` is the Bond number encoding surface
tension `T`, gravity `g`, and wavelength `λ`, and `y(u)` is the free-surface elevation in
conformal coordinates.

**Why a rigorous residual bound?** Formal perturbation theory gives approximate solutions
(e.g., truncated Stokes expansions) but no rigorous certificate of how good the approximation is.
A rigorous residual bound is the **first half** of a computer-assisted existence proof: it
provides the quantitative `η` in `‖R‖ ≤ η` that a downstream Newton–Kantorovich argument
consumes. For gravity-capillary waves the parameter space `(μ, β)` has a rich bifurcation
structure where purely analytic arguments are incomplete, so even a residual bound (without the
existence upgrade) is of independent value as a quantitative diagnostic.

**What the residual bound delivers, and what it doesn't:**
- ✅ A certified explicit constant `η` such that `‖R‖∞ ≤ η`, for the user's chosen
  `(a_coeffs, μ, β)`.
- ✅ Lean-side guarantee that `η` is free of unverified numerical claims (all heavy arithmetic
  is in Arb interval-ball form on the Julia side; Lean checks the universal analytic lemmas).
- ❌ **Not** included: any bound on the inverse linearized operator `‖(DT)^{-1}‖`. That step
  would need extra Arb-computed certificate data (specifically, an operator-norm bound on
  the inverse of the linearisation), plus a paper-level analysis of the Babenko operator's
  spectrum — neither of which is part of this project.
- ❌ **Not** included: a Lean-side fixed-point argument upgrading `‖R‖∞ ≤ η` to
  `∃ y_star, R(y_star) = 0`. That is the Newton–Kantorovich step proper, and is left for a
  follow-up.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Input                                                               │
│  a_1 … a_N  (cosine coefficients),  μ,  β  (Float64 / exact ratl.) │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                ┌───────────▼──────────────┐
                │   BabenkoRigorous.jl     │   Julia / Arblib.jl
                │   (ball arithmetic)      │   (FLINT / Arb)
                │                          │
                │  • Path A: H on trig     │
                │    polynomials (exact)   │
                │  • Path B: H on G via    │
                │    DFT + aliasing bound  │
                │  • TaylorSeries assembly │
                │    of capillary term     │
                │  • ‖·‖∞ bound on M_grid │
                └───────────┬──────────────┘
                            │  outputs certificate
                            │  (ε₀, L, DLower, M_grid, …)
                            ▼
                ┌───────────────────────────┐
                │   Certificate.lean        │   Lean 4 source (auto-gen)
                │   (literal numerics as    │
                │    Lean hypotheses)       │
                └───────────┬───────────────┘
                            │
                ┌───────────▼───────────────┐
                │   BabenkoWave (Lean 4)    │
                │                           │
                │  AliasingErrorBound.lean  │   bound on DFT aliasing
                │  TaylorSeries.lean        │   Taylor remainder logic
                │  HilbertTrig.lean         │   Path A: H on trig polys
                │  HilbertDFT.lean          │   Path B: H via DFT
                │  Residual.lean            │   assembles ‖R‖∞ ≤ ε₀+…  │
                └───────────────────────────┘
```

**TaylorSeries flow** (for the capillary term `H[∂_u(G)]` where `G = (1+H[y_u])/D`):

```
G at M_grid points  →  DFT  →  H[G] at M_grid points
         ↓ differentiate (× ik in frequency)
G^{(m)} at M_grid points  →  DFT  →  H[G^{(m)}] = (H[G])^{(m)}
         ↓
TaylorSeries with ball coefficients + aliasing error from Path B
         ↓
sup-norm bound via subinterval evaluation + Taylor remainder
```

---

## 4. The Two H Pathways

The Hilbert transform appears in four places in the residual. The treatment differs depending
on whether the input is a trig polynomial (exact) or a nonlinear function (approximated by DFT).

### Path A — H on trig polynomials (exact)

**Rule:** `H[cos(ku)] = sin(ku)`, `H[sin(ku)] = -cos(ku)`, `H[1] = 0`.

Applied to:
- `H[y_u]` — since `y = Σ a_k cos(ku)`, we have `y_u = -Σ k a_k sin(ku)`, so
  `H[y_u] = Σ k a_k cos(ku)`. Exact, no error.

**Lean treatment:** `HilbertOnTrigPoly.lean` proves the mode-by-mode action as a lemma; no
aliasing error term.

### Path B — H via certified DFT on non-trig-polynomial input

**Used for:**
- `H[y · y_u]` — *mathematically* a trig polynomial of degree `≤ 2N`, but the implemented
  pipeline holds `y · y_u` as a `TaylorSeries` (product of two `TaylorSeries` objects) and
  routes `H` through `fft_hilbert_series` for uniformity with the capillary path.
- `H[∂_u G]` — `G = (1+H[y_u])/D` is *not* a trig polynomial (`D = √(...)` has infinitely many
  Fourier modes), so Path B is genuinely required.

**Procedure (`fft_hilbert_series`):**
1. Evaluate input `f` on an `N`-point uniform grid as `Vector{Arb}`.
2. Compute the circular convolution `h ⋆ f` via `AcbPoly` multiplication (FLINT's certified FFT).
   The kernel `h_n = (2/N) cot(πn/N)` for odd `n` has DFT exactly equal to the spectral
   Hilbert multiplier `−i · sign(k)`, Nyquist zeroed.
3. Add a rigorous aliasing radius to every output ball:
   `ε_alias = 4·Cp / ((p−1)·(N/2−1)^{p−1})` for `p ≥ 2`, where `Cp ≥ ‖f^{(p)}‖∞`.
   The factor 4 is `(multiplier disagreement ≤ 2) × (real-Fourier symmetry × 2)`; the
   denominator comes from the integral test `Σ_{m ≥ M} m^{−p} ≤ (M−1)^{1−p}/(p−1)` with `M = N/2`.

**Output `TaylorSeries.sup_bounds`:** the output TaylorSeries' `sup_bounds[m+1]` is set to
`‖G^{(m+2)}‖∞ · π²/3` (Fourier-tail decay: two integrations by parts on `c_k(G^{(m)})` give
`|c_k| ≤ ‖G^{(m+2)}‖∞ / k²`; summing yields `π²/3`). This requires the caller to supply two
extra-margin bounds (`Gpp_bound` on `‖G^{(M_order+2)}‖∞` and `Gppp_bound` on `‖G^{(M_order+3)}‖∞`)
beyond the input series' own `sup_bounds`.

**Key identity used:** `H` commutes with `∂_u`, so `(H[f])^{(m)} = H[f^{(m)}]`. The TaylorSeries
of `H[f]` is built by applying `fft_hilbert` at each derivative level `m = 0, …, M_order`.

**Lean treatment:** `AliasingErrorBound.lean` provides the key inequality; `HilbertDFT.lean`
packages the DFT Hilbert transform on interval-valued grid data.

---

## 5. Julia Component

The Julia package `BabenkoRigorous.jl` performs all numerics. It uses
[Arblib.jl](https://github.com/kalmarek/Arblib.jl) throughout: every floating-point operation
produces a ball `[midpoint ± radius]` that is guaranteed to contain the true value.

**Outputs (the certificate):**

| Symbol | Meaning |
|--------|---------|
| `ε₀` | Arb upper bound on `‖R‖∞` from `sup_bound(ts_R)` (Taylor-remainder form, not pure Lipschitz) |
| `sup_bounds_R` | Per-derivative supnorm vector for `R` (length `M_order+2`), used by Lean's `grid_to_supNorm_taylor` |
| `DLower` | Arb lower bound on `D = √(y_u²+(1+H[y_u])²)` (needed for capillary term) |
| `N_fft` | DFT grid size used in Path B `fft_hilbert_series` |
| `M_order` | Taylor expansion order |
| `Gpp_bound`, `Gppp_bound` | Two extra-margin derivative bounds per Path-B Hilbert call (`y·y_u` for β=0; `∂_u G` for β≠0) |

**File table (current):**

| File | Role |
|------|------|
| `src/BabenkoRigorous.jl` | Package entry point |
| `src/FourierSeries.jl` | `TrigPoly` struct + exact `hilbert`, `deriv`, `tp_eval`, `supnorm_bound` (Path A) |
| `src/TaylorSeries.jl` | `TaylorSeries` struct (sup_bounds vector); `*` (exact Leibniz), `inv_series`/`sqrt_series` (Faà di Bruno via `_faa_di_bruno_derivs`), `deriv_series`, `truncate_order`, `dmin_bound`, `trigpoly_to_taylor`, `sup_bound`/`inf_bound` |
| `src/GDerivatives.jl` | `compute_G_taylor`: certified TaylorSeries for `G = (1+H[y_u])/D` |
| `src/FFTHilbert.jl` | Path B: certified discrete Hilbert via `AcbPoly` convolution; `fft_hilbert` (per-grid + aliasing) and `fft_hilbert_series` (TaylorSeries lift with Fourier-tail `sup_bounds`) |
| `src/ResidualBound.jl` | Assembles all terms; `residual_bound(y, μ, β; M_order, N_grid, prec)` returns `sup_bound(ts_R)` |
| `scripts/m1_plots.jl` … `m5_plots.jl` | Per-milestone diagnostics + figures |

`src/Certificate.jl` (Lean-readable rational output) exists and produced the
shipped `WaveCAP/Certificate.lean` (N=65536 run). Note its emitter still
writes the retired monolithic-axiom format — see Q1 in
`audit_2026-07-03.md` before regenerating.

---

## 6. Lean Component

The Lean 4 package `BabenkoWave` imports the auto-generated `Certificate.lean` as a set of
hypotheses and proves `‖R‖∞ ≤ ε₀ + L·π/M_grid` by pure logic. No Lean code executes
floating-point arithmetic; all numeric claims are asserted as hypotheses coming from Julia.

**One paragraph:** The Lean proof is organized as a short chain: `AliasingErrorBound.lean`
establishes the DFT aliasing inequality in the abstract (for any function in the analytic strip);
`HilbertTrig.lean` records the exact action of `H` on cosines and sines; `TaylorSeries.lean`
provides the Taylor remainder bound used to propagate Path B grid evaluations to all of
`[0, 2π]`; and `Residual.lean` combines these with the certificate hypotheses to close the
`‖R‖∞ ≤ ε` claim.

**File table (planned):**

| File | Proves |
|------|--------|
| `BabenkoWave/AliasingErrorBound.lean` | `|H[G](u) − H_M[G](u)| ≤ C e^{−ρ M}` for all `u` |
| `BabenkoWave/HilbertTrig.lean` | `H[cos(ku)] = sin(ku)` etc. as Lean lemmas |
| `BabenkoWave/HilbertDFT.lean` | Correctness of DFT Hilbert multiplier on band-limited data |
| `BabenkoWave/TaylorSeries.lean` | Taylor remainder bound for analytic functions on `[0,2π]` |
| `BabenkoWave/Residual.lean` | Main theorem: `‖R‖∞ ≤ ε` from certificate hypotheses |
| `BabenkoWave/Certificate.lean` | Auto-generated; contains `ε₀, L, DLower, M_grid` as `def` |

---

## 7. The Certificate

Julia produces a file `Certificate.lean` that looks roughly like:

```lean
-- Auto-generated by BabenkoRigorous.jl — do not edit
def epsilon0 : Float := 3.14e-6   -- Arb upper bound on grid residual
def L        : Float := 0.042     -- Lipschitz constant
def DLower   : Float := 0.997     -- lower bound on arc-length factor D
def M_grid   : ℕ    := 4096       -- DFT grid size
```

These become hypotheses in `Residual.lean`. The final theorem statement is:

```lean
theorem residual_bound : ‖R‖_∞ ≤ epsilon0 + L * π / M_grid := by
  ...  -- combines AliasingErrorBound + TaylorSeries + grid evaluation
```

**Trust argument:** Julia's Arblib computations are certified by the FLINT/Arb library, which is
a well-audited C library for rigorous real arithmetic. Lean does not re-check the Arb
arithmetic; it trusts the certificate values as axioms. The trust chain is therefore:

```
Lean kernel  +  Mathlib  →  trust at the logical level
FLINT/Arb              →  trust at the arithmetic level
```

This is the same division of trust used in virtually all computer-assisted proofs that combine
interval arithmetic with a proof assistant.

---

## 8. Comparison to cap-ccf

| Aspect | cap-ccf (CCF blow-up) | This project (Babenko waves) |
|--------|----------------------|------------------------------|
| Domain | `ℝ` (non-periodic) | `[0, 2π]` (periodic) |
| Basis functions | Hermite functions `h_n(z)` | Cosines `cos(ku)` |
| Hilbert transform | Composite: standard piece + smooth FFT kernel correction | Two clean paths: exact trig (Path A) + periodic DFT (Path B) |
| Aliasing | Euler-Maclaurin discretization error for kernel integrals | Standard Fourier tail decay (`e^{-ρ M}`) |
| Lean aliasing file | `CompositeTrapezoid.lean` | `AliasingErrorBound.lean` (analogous role) |
| TaylorSeries struct | `taylor_series.jl` in cap-ccf | Adapted directly |
| Gaussian moments | Needed (Hermite weight) | Not needed |
| Operator | Nonlinear integro-differential (CCF, non-local) | Nonlinear integro-differential (Babenko, also non-local) |
| Certificate format | Implicit in scripts | Explicit `Certificate.lean` file |

**What is reused from cap-ccf:**
- The `TaylorSeries` struct pattern and Taylor remainder logic.
- The general two-phase architecture (rigorous arithmetic in Julia → logical scaffolding in Lean).
- Arblib usage patterns (`arblib_guide.md` is directly applicable).

**What is new or simpler:**
- The periodic domain eliminates the smooth kernel correction needed in cap-ccf.
- Path A (exact H on trig polynomials) has no analogue in cap-ccf.
- The certificate boundary between Julia and Lean is made explicit via `Certificate.lean`.
- No Hermite function recurrence; no Gaussian moment bounds.

---

## 9. Staging Plan

1. **Float64 prototype** ✓ — `babenko_residual.jl` + `hilbert.jl` confirm the equation is
   correctly implemented and give expected residual magnitudes for test cases.

2. **Arb scalar evaluation** — Port `babenko_residual.jl` to Arblib.jl. Verify that the ball
   arithmetic enclosures are tight at the same test cases. This validates the basic pipeline
   before tackling the Hilbert transform.

3. **Path A implementation** — Implement `hilbert_trig.jl`: exact H on trig polynomials via
   coefficient vector convolution in Arb. Unit-test against Float64 results.

4. **Path B prototype (no aliasing)** — Implement DFT Hilbert transform on Arb grid data.
   Compare to Path A on a trig polynomial input where the answer is known exactly.

5. **Aliasing error bound** — Implement `aliasing.jl`: given a lower bound on the analytic
   strip width `ρ` and the grid size `M_grid`, produce the Arb aliasing error. Unit-test on a
   function with known Fourier tail decay.

6. **Full residual in Arb** — Assemble `residual.jl`: compute all six terms of the Babenko
   equation in Arb, combine aliasing errors, return `ε₀`, `L`, `DLower`, `M_grid`.

7. **Certificate serialization** — Implement `certificate.jl` to write `Certificate.lean`.
   Verify the file parses correctly in Lean 4 / Mathlib.

8. **Lean scaffolding** — Write `HilbertTrig.lean` (should be short, essentially just the
   three mode-action lemmas) and a stub `Residual.lean` that accepts the certificate as
   hypotheses and asserts the final bound.

9. **AliasingErrorBound.lean** — Prove the aliasing inequality in Lean using Mathlib's Fourier
   series and analytic function machinery. This is the most mathematically substantial Lean file,
   analogous to `CompositeTrapezoid.lean` in cap-ccf.

10. **TaylorSeries.lean** — Prove the Taylor remainder bound for analytic functions on `[0,2π]`.
    Likely lifts from existing Mathlib lemmas.

11. **Close the proof** — Fill in `Residual.lean` to deduce `‖R‖∞ ≤ ε₀ + L·π/M_grid` from
    the assembled lemmas and certificate hypotheses.

12. **Run and tune** — Run the full pipeline on the small-amplitude Stokes wave test case.
    Adjust `N`, `M_grid`, `taylor_order`, and `PREC` until `ε` is small enough for the
    contraction argument.

The ordering prioritizes getting a closed proof on a simple case before optimizing the numerics.
Steps 8–10 can proceed in parallel with steps 4–6.

---

## 10. Open Questions

**Mathematical:**

- **Is the approximate solution good enough?** The Float64 residuals (‖R‖∞ ≈ 1.1e-2 for ε=0.15
  Stokes, 3.7e-3 for the linear gravity-capillary mode) are encouragingly small, but the Arb
  ball inflation may widen them. A back-of-envelope estimate is needed for how much Arb overhead
  to expect vs. grid size `M_grid`.

- **Analyticity strip width `ρ`:** The aliasing bound decays as `e^{-ρ M_grid}`. What is the
  actual analyticity strip for `G = (1+H[y_u])/D` near a given approximate solution? This
  determines the minimum `M_grid` needed for the aliasing error to not dominate.

- **`DLower` positivity:** The capillary term involves division by `D`. We need a rigorously
  certified lower bound `DLower > 0`. This requires the approximate solution to be "well-posed"
  in the sense that the free surface is non-self-intersecting. For small-amplitude waves this
  holds easily; near the Wilton ripple resonances or large amplitudes it needs checking.

- **Contraction constant:** Once ‖R‖∞ ≤ ε is proved, a Newton-Kantorovich argument also needs
  bounds on ‖DΦ^{-1}‖ (the inverse linearized operator). That step is not addressed here but
  shapes the required size of ε.

**Implementation:**

- **Choice of `M_grid` and `taylor_order`:** These are the main tuning parameters. Too small
  → aliasing / Taylor error dominates. Too large → Arb computation becomes slow. A sweep script
  should be written early to understand the tradeoff.

- **Lean Fourier analysis imports:** `AliasingErrorBound.lean` will need Mathlib's
  `MeasureTheory.Function.FourierSeries` and analytic function results. These are present in
  Mathlib but their exact API needs checking against the current Mathlib4 version.

- **Certificate format stability:** If Lean changes how `Float` literals are parsed, the
  auto-generated `Certificate.lean` could break. Consider using rational literals (`p/q : ℚ`)
  or explicit interval endpoints instead of `Float` for the certificate values.

- **Lean 4 version pinning:** The proof must compile against a specific `lean-toolchain` and
  Mathlib commit. This should be decided before writing any Lean files.

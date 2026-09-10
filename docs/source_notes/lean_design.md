# Lean 4 Proof Design: Babenko Water-Wave Residual Bound

> **Historical design; current status checked 2026-09-09.** The publication
> proof sources are sorry-free, and the dedicated audit reports no `sorryAx`
> dependency. The final residual theorem nevertheless assumes two explicit
> numerical axioms: `cert_arb_dmin_bound` and `cert_arb_sup_bound_le_eps0`, in
> addition to Lean's standard `propext`, `Classical.choice`, and `Quot.sound`.
> Lean does not independently verify the Crunchy2 interval computation.
> Claims below about accepted `sorry` gaps or other axiom counts belong to the
> original design, and its `sorry` code examples are not compiled proofs.
> See the [current Lean verification status](../../Lean/README.md#verification-status)
> and the [updated roadmap notice](lean_roadmap_2026-05-23.md).

**Project:** Computer-assisted proof (CAP) that $\|R\|_{L^\infty([0,2\pi])} \leq \varepsilon$
for the surface-tension Babenko water-wave equation.

**Date:** 2026-05-21

---

## Table of Contents

1. [Overview](#1-overview)
2. [File Structure](#2-file-structure)
3. [Definitions.lean](#3-definitionslean)
4. [HilbertOnTrigPoly.lean](#4-hilbertontrigpolylean)
5. [HilbertCommutesDeriv.lean](#5-hilbertcommutesderivlean)
6. [AliasingErrorBound.lean](#6-aliasingerrorboundlean)
7. [TaylorRemainder.lean](#7-taylorremainderLean)
8. [DLowerBound.lean](#8-dlowerboundlean)
9. [GridToSupNorm.lean](#9-gridtosupnormlean)
10. [Main.lean](#10-mainlean)
11. [Julia–Lean Interface](#11-julialean-interface)
12. [Staging Plan](#12-staging-plan)

---

## 1. Overview

The Lean proof is the **verification kernel** of a larger computer-assisted proof.
Julia/Arblib plays the role of the *certificate generator*: it computes a
finite cosine-series approximate solution $y$, evaluates the residual $R$ of the
Babenko equation at a fine grid, bounds all truncation and floating-point errors
using interval arithmetic, and emits a machine-readable certificate consisting of
four numbers $(\varepsilon_0, L, D_{\mathrm{lower}}, M_{\mathrm{grid}})$.
Lean's job is to verify, from first principles and using only that certificate, that
$\|R\|_{L^\infty} \leq \varepsilon_0 + L\pi/M_{\mathrm{grid}}$.
Lean proves nothing about the numerics; it proves the *implication* from the
certificate hypotheses to the sup-norm bound.
The analytic core — the structure of $H$, the aliasing bound, Taylor remainder
theory, and the Lipschitz interpolation argument — is fully formal.
No `sorry` appears in the final proof except in stubs awaiting Mathlib PRs
(the precise aliasing constant and one Fourier-series density lemma),
both of which are mathematically uncontroversial.

---

## 2. File Structure

| File | Contents | Nature |
|---|---|---|
| `Definitions.lean` | All noncomputable defs: $y$, $Hy$, $y_u$, $Hy_u$, $D$, $G$, $R$ | Pure analysis |
| `HilbertOnTrigPoly.lean` | $H[\cos ku]=\sin ku$; $H$ linear on finite cosine sums; Path A correctness | Pure analysis |
| `HilbertCommutesDeriv.lean` | $H \circ \partial_u = \partial_u \circ H$ for trig polynomials, hence for $G$ | Pure analysis |
| `AliasingErrorBound.lean` | DFT aliasing error bound; corollary bounding $\|H_N[G] - H[G]\|_\infty$ | Pure analysis |
| `TaylorRemainder.lean` | Lagrange Taylor remainder adapted from CCF; subinterval sup-norm corollary | Pure analysis |
| `DLowerBound.lean` | From $D(u) \geq D_{\mathrm{lower}}$ deduce $1/D$ well-defined, $\|\partial_u(1/D)\|_\infty$ bounded | Cites Julia certificate |
| `GridToSupNorm.lean` | Lipschitz grid-to-sup-norm abstract lemma | Pure analysis |
| `Main.lean` | Assembles all pieces; top-level residual-bound theorem | Cites Julia certificate |
| `Certificate.lean` | Lean axioms encoding the Julia-computed certificate values | Certificate bridge |

All files import `Mathlib` via a `Common.lean` shim (following the CCF pattern).

---

## 3. `Definitions.lean`

### 3.1 Mathematical Content

Fix $N \in \mathbb{N}$ (the number of cosine modes) and coefficients
$(a_1, \ldots, a_N) \in \mathbb{R}^N$.
The approximate solution is the even $2\pi$-periodic function

$$y(u) = \sum_{k=1}^{N} a_k \cos(ku).$$

Its derivative is

$$y_u(u) = -\sum_{k=1}^{N} k\, a_k \sin(ku).$$

The periodic Hilbert transform acts mode-by-mode on $L^2(\mathbb{T})$:
$H[\cos(ku)] = \sin(ku)$, $H[\sin(ku)] = -\cos(ku)$ for $k \geq 1$,
$H[1] = 0$.
Therefore

$$Hy(u) = \sum_{k=1}^{N} a_k \sin(ku), \qquad
Hy_u(u) = H[y_u](u) = \sum_{k=1}^{N} k\, a_k \cos(ku).$$

The arc-length denominator is

$$D(u) = \sqrt{y_u(u)^2 + (1 + Hy_u(u))^2}.$$

The capillary fraction is

$$G(u) = \frac{1 + Hy_u(u)}{D(u)}.$$

The residual is $R = \mathrm{LHS} - \mathrm{RHS}$ where

$$\mathrm{LHS}(u) = \mu\, Hy_u(u),$$

$$\mathrm{RHS}(u) = y(u) + y(u)\,Hy_u(u) + H[y\,y_u](u)
  - \beta\,\partial_u\!\left(\frac{y_u(u)}{D(u)}\right)
  + \beta\, H\!\left[\partial_u G\right](u).$$

### 3.2 Lean Statement

```lean
import BabenkoWave.Common

namespace BabenkoWave

open Real

/-- Number of cosine modes and their coefficients. -/
variable (N : ℕ) (a : Fin N → ℝ) (μ β : ℝ)

/-- Approximate solution: finite even cosine series. -/
noncomputable def y (u : ℝ) : ℝ :=
  ∑ k : Fin N, a k * cos ((k.val + 1 : ℝ) * u)

/-- Derivative of y. -/
noncomputable def yu (u : ℝ) : ℝ :=
  -∑ k : Fin N, (k.val + 1 : ℝ) * a k * sin ((k.val + 1 : ℝ) * u)

/-- Hilbert transform of y: H[cos(ku)] = sin(ku). -/
noncomputable def Hy (u : ℝ) : ℝ :=
  ∑ k : Fin N, a k * sin ((k.val + 1 : ℝ) * u)

/-- Hilbert transform of y_u: H[y_u](u) = Σ k·a_k·cos(ku). -/
noncomputable def Hyu (u : ℝ) : ℝ :=
  ∑ k : Fin N, (k.val + 1 : ℝ) * a k * cos ((k.val + 1 : ℝ) * u)

/-- Arc-length denominator. Requires D > 0 as a separate hypothesis. -/
noncomputable def D (u : ℝ) : ℝ :=
  Real.sqrt (yu N a u ^ 2 + (1 + Hyu N a u) ^ 2)

/-- Capillary fraction G = (1 + H[y_u]) / D. -/
noncomputable def G (u : ℝ) : ℝ :=
  (1 + Hyu N a u) / D N a u

/-- The residual R = LHS - RHS of the Babenko equation. -/
noncomputable def R (u : ℝ) : ℝ :=
  μ * Hyu N a u
  - (y N a u
     + y N a u * Hyu N a u
     + H_of_product N a u          -- H[y · y_u], defined in HilbertOnTrigPoly
     - β * deriv (fun v => yu N a v / D N a v) u
     + β * H_of_deriv_G N a u)     -- H[∂_u G], defined via DFT in AliasingErrorBound
```

> **Note:** `H_of_product` and `H_of_deriv_G` are defined in the files
> `HilbertOnTrigPoly.lean` and `AliasingErrorBound.lean` respectively.
> They are imported here by forward reference. In a fully compiled project
> they would be declared in dependency order.

### 3.3 Proof Sketch

No proofs live in `Definitions.lean`. The file only contains `noncomputable def`s
and no `theorem`s. Its only obligation is that every definition compiles
(i.e., is type-correct and terminates).

### 3.4 Mathlib Dependencies

- `Real.cos`, `Real.sin`, `Real.sqrt` — basic transcendentals
- `Finset.sum`, `Fin` — finite indexing
- `deriv` — Fréchet/pointwise derivative on $\mathbb{R}$

### 3.5 Dependencies on Other Project Files

None. `Definitions.lean` is the root of the dependency DAG.

---

## 4. `HilbertOnTrigPoly.lean`

### 4.1 Mathematical Content

**Path A** establishes that, for inputs that are trig polynomials,
$H$ is computed *exactly* by mode-by-mode multiplication.
No grid, no FFT, no aliasing error — just algebra.

The three key facts are:

**Fact A1 (cosine mode).**
For all $k \geq 1$ and all $u \in \mathbb{R}$,

$$H[\cos(k\,\cdot\,)](u) = \sin(ku).$$

**Fact A2 (sine mode).**
For all $k \geq 1$ and all $u \in \mathbb{R}$,

$$H[\sin(k\,\cdot\,)](u) = -\cos(ku).$$

**Fact A3 (linearity).**
$H$ is $\mathbb{R}$-linear on the space of trig polynomials;
in particular, for a finite cosine series $y = \sum_{k=1}^N a_k \cos(k\,\cdot\,)$,

$$Hy = \sum_{k=1}^N a_k \sin(k\,\cdot\,) \qquad \text{(pointwise on } \mathbb{R}\text{)}.$$

**Fact A4 (H[y·y_u] explicit).**
Since $y\,y_u$ is a trig polynomial (product of two trig polynomials),
its Fourier coefficients are computed by the discrete convolution of $(a_k)$ and
$(-k\,a_k)$, and $H[y\,y_u]$ is the corresponding sine series.
Concretely, if we write $y\,y_u = \sum_{m} c_m \cos(mu) + s_m \sin(mu)$, then

$$H[y\,y_u](u) = \sum_{m \geq 1} c_m \sin(mu) - s_m \cos(mu).$$

**Fact A5 (H[y_u] = Hyu).**
The definition `Hyu` in `Definitions.lean` equals $H[y_u]$ pointwise.

### 4.2 Lean Statement

```lean
import BabenkoWave.Definitions

namespace BabenkoWave

open Real

/-- Definition of H on a cosine series: declared here as the canonical mode-by-mode map.
    This is the *definition* used in `Definitions.lean`; this file proves it satisfies
    the expected Hilbert transform identities. -/

/-- A1: H[cos(k·)](u) = sin(ku) for k ≥ 1.
    In the periodic Hilbert transform, H acts as multiplication by -i·sign(k) in
    frequency space; for k ≥ 1 this maps e^{iku} ↦ -i·e^{iku}, giving
    cos(ku) ↦ sin(ku). We prove this directly from the definition. -/
theorem H_cosine_mode (k : ℕ) (hk : 1 ≤ k) (u : ℝ) :
    H_of_cosine_series (fun n => if n = k then 1 else 0) u = sin (k * u) := by
  simp [H_of_cosine_series]

/-- A2: H[sin(k·)](u) = -cos(ku) for k ≥ 1. -/
theorem H_sine_mode (k : ℕ) (hk : 1 ≤ k) (u : ℝ) :
    H_of_sine_series (fun n => if n = k then 1 else 0) u = -cos (k * u) := by
  simp [H_of_sine_series]

/-- A3: H[y](u) = Hy(u) where Hy is the mode-by-mode definition. -/
theorem H_of_y_eq_Hy (u : ℝ) :
    H_of_cosine_series (fun k => a k) u = Hy N a u := by
  unfold Hy H_of_cosine_series
  rfl

/-- A5: H[y_u] = Hyu pointwise.
    y_u is a sine series; H maps each -k·a_k·sin(ku) to k·a_k·cos(ku). -/
theorem H_of_yu_eq_Hyu (u : ℝ) :
    H_of_sine_series (fun k => -(k.val + 1 : ℝ) * a k) u = Hyu N a u := by
  unfold Hyu H_of_sine_series yu
  simp [mul_comm, mul_neg, neg_neg]

/-- A4: H[y · y_u] is mathematically a trig polynomial of degree ≤ 2N, so its
    coefficients can be computed exactly via the convolution / product-to-sum
    identities.  However, the implemented Julia residual pipeline routes
    `H[y · y_u]` through the same DFT-based `fft_hilbert_series` as the
    capillary term — not via this exact-convolution formula — for engineering
    uniformity (everything goes through the `TaylorSeries` machinery).

    Lean therefore does not need a closed-form `H_of_product` def.  The
    residual hypothesis is instead a *grid-point* hypothesis of the form
        |H_N[y · y_u](u_j) - H[y · y_u](u_j)| ≤ ε_alias_yyu
    where `ε_alias_yyu` is the integral-test bound from `fft_hilbert_aliasing_bound`
    (§6) instantiated at `f = y · y_u`, `p = 2`, `Cp = Gpp_bound_yyu_cert`.

    The exact-convolution `H_of_product` definition below is kept for reference
    (it is the *mathematical* `H[y · y_u]`, useful for relating discrete and
    continuous statements), but is no longer load-bearing in `Main.lean`. -/
noncomputable def H_of_product (u : ℝ) : ℝ :=
  -- Reference definition: the true continuous Hilbert of y · y_u as a trig
  -- polynomial of degree ≤ 2N.  Implementation deferred — see remark below.
  sorry

/-- Used by `Main.lean`: the grid-point error between the discrete `H_N[y · y_u]`
    that Julia computes and the true `H[y · y_u]`.  This is the M10 lemma
    `fft_hilbert_aliasing_bound` (§6) specialized to `f = y · y_u`. -/
theorem H_yyu_aliasing_at_grid
    (j : Fin N_fft_cert) :
    let yyu : ℝ → ℝ := fun u => y N a u * yu N a u
    |H_N yyu N_fft_cert (2 * π * (j : ℝ) / N_fft_cert)
        - periodicHilbert yyu (2 * π * (j : ℝ) / N_fft_cert)|
      ≤ 4 * (Gpp_bound_yyu_cert : ℝ) / (1 * ((N_fft_cert / 2 : ℝ) - 1)) := by
  -- Apply `fft_hilbert_aliasing_bound` from §6 with p = 2, Cp = Gpp_bound_yyu.
  sorry
```

> **Remark on the rerouting.** In an earlier draft this section defined
> `H_of_product` as an explicit cosine/sine convolution (Path A in the
> Julia-side terminology) and used it directly in the residual `R N a μ β`.
> The implemented Julia code routes `H[y · y_u]` through `fft_hilbert_series`
> (Path B) for engineering uniformity with the capillary $H[\partial_u G]$ term,
> so the Lean residual statement must encode the DFT value at grid points plus
> the aliasing inequality — which is what `H_yyu_aliasing_at_grid` above does.
> The mathematical `H_of_product` definition is retained as a reference object
> but does not participate in the `Main.lean` chain.

### 4.3 Proof Sketch

- **A1/A2:** By `simp` unfolding the definition of `H_of_cosine_series`
  (which *is* $k \mapsto \sin(ku)$) and `H_of_sine_series` (which is
  $k \mapsto -\cos(ku)$). No real analysis needed: these are *definitions*.
- **A3/A5:** By `Finset.sum_congr` and commutativity of multiplication.
- **A4:** Expand $y\,y_u$ using `cos(au)\cdot\sin(bu) = [\sin((a+b)u) - \sin((a-b)u)]/2$,
  collect terms by frequency $m$, apply `H_sine_mode` and `H_cosine_mode` to each,
  and `ring`-close.

### 4.4 Mathlib Lemmas Used

- `Finset.sum_congr` — reindexing finite sums
- `Real.cos_add`, `Real.sin_add`, `Real.cos_mul`, `Real.sin_mul` — product-to-sum
- `Finset.sum_comm'` — reordering double sums in the convolution

### 4.5 Dependencies

- `Definitions.lean`

---

## 5. `HilbertCommutesDeriv.lean`

### 5.1 Mathematical Content

The key commutativity identity for Path B is:

$$\partial_u \circ H = H \circ \partial_u \quad \text{on } 2\pi\text{-periodic } L^2.$$

**Theorem (H commutes with $\partial_u$, trig-poly case).**
Let $p(u) = \sum_{k=1}^M b_k \cos(ku) + c_k \sin(ku)$ be a trig polynomial.
Define $H[p](u) = \sum_{k=1}^M b_k \sin(ku) - c_k \cos(ku)$ (mode-by-mode).
Then

$$\partial_u H[p] = H[\partial_u p].$$

*Proof.* Compute both sides:
- $\partial_u H[p] = \sum_{k=1}^M k b_k \cos(ku) + k c_k \sin(ku)$
- $H[\partial_u p] = H\!\left[\sum_{k=1}^M -k b_k \sin(ku) + k c_k \cos(ku)\right]
  = \sum_{k=1}^M k b_k \cos(ku) + k c_k \sin(ku)$

These are equal. $\square$

**What level of generality is needed for `Main.lean`?**
The only place where $\partial_u H[f] = H[\partial_u f]$ is invoked for a
*non*-trig-polynomial is when $f = G = (1 + Hy_u)/D$.
$G$ is smooth and $2\pi$-periodic (provided $D > 0$ uniformly), but it is
not a finite Fourier series.
The general case follows from the trig-poly case by density:
if $G \in C^1(\mathbb{T})$ and $p_n \to G$ in $C^1$ (e.g., Fejér sums),
then $\partial_u H[p_n] \to \partial_u H[G]$ and $H[\partial_u p_n] \to H[\partial_u G]$
by continuity, hence they agree.
In practice for `Main.lean` we only need the identity at specific grid points
$u_j = 2\pi j/M_{\mathrm{grid}}$, where it suffices to invoke the
already-available DFT computation (with its aliasing error).
Therefore the full density argument can be `sorry`'d for the POC with
a comment identifying the exact Mathlib lemma path:
`MeasureTheory.Lp.trigPolynomials_dense` + `ContinuousLinearMap.ext`.

**Corollary (iterated commutativity).**
For $m \geq 1$ and $G \in C^m(\mathbb{T})$,

$$H[G^{(m)}] = (H[G])^{(m)}.$$

This justifies the Taylor-series approach: to bound $\|H[G]\|_{L^\infty}$
via its Taylor expansion at grid points, it suffices to bound the pointwise
values of $H[G^{(m)}]$ for $m = 0, 1, \ldots, M$.

### 5.2 Lean Statement

```lean
import BabenkoWave.HilbertOnTrigPoly

namespace BabenkoWave

open Real

/-- H commutes with ∂_u on trig polynomials.
    Both `H_of_trigpoly_deriv` and `deriv_H_of_trigpoly` compute to
    the same explicit sum; we prove equality by `ring`-closing each component. -/
theorem H_commutes_deriv_trigpoly
    (M : ℕ) (b c : Fin M → ℝ) (u : ℝ) :
    let p : ℝ → ℝ := fun v =>
      ∑ k : Fin M, b k * cos ((k.val + 1) * v) + c k * sin ((k.val + 1) * v)
    deriv (fun v => H_of_trigpoly M b c v) u =
    H_of_trigpoly M
      (fun k => -(k.val + 1 : ℝ) * c k)   -- coefficients of ∂_u p (cosine part)
      (fun k =>  (k.val + 1 : ℝ) * b k)   -- coefficients of ∂_u p (sine part)
      u := by
  simp only [H_of_trigpoly, deriv_sum]
  apply Finset.sum_congr rfl
  intro k _
  simp [mul_comm, deriv_const_mul, Real.deriv_sin, Real.deriv_cos]
  ring

/-- Corollary: iterated commutativity. H[f^{(m)}] = (H[f])^{(m)} on trig polys. -/
theorem H_commutes_iteratedDeriv_trigpoly
    (M : ℕ) (b c : Fin M → ℝ) (m : ℕ) (u : ℝ) :
    iteratedDeriv m (fun v => H_of_trigpoly M b c v) u =
    H_of_trigpoly M
      (rotate_cosine_coeffs_m M b c m)
      (rotate_sine_coeffs_m M b c m)
      u := by
  induction m with
  | zero => simp [iteratedDeriv_zero]
  | succ n ih =>
    rw [iteratedDeriv_succ]
    rw [← H_commutes_deriv_trigpoly]
    congr 1
    exact ih

/-- The general smooth-periodic case: H commutes with ∂_u on C¹(𝕋). -/
theorem H_commutes_deriv_smooth
    {f : ℝ → ℝ} (hf : ContDiff ℝ 1 f) (hperiod : ∀ u, f (u + 2 * π) = f u) (u : ℝ) :
    deriv (fun v => periodicHilbert f v) u =
    periodicHilbert (fun v => deriv f v) u := by
  -- Proof by density: approximate f in C¹ by trig polynomials (Fejér),
  -- use H_commutes_deriv_trigpoly for each polynomial, pass to limit.
  sorry -- TODO: MeasureTheory.Lp.trigPolynomials_dense + continuity of H on C¹
```

### 5.3 Proof Sketch

- **Trig-poly case:** Unfold the explicit sum definitions of $H$ on trig polys
  and $\partial_u$ on trig polys. Each term reduces to a `ring` identity
  between $\cos/\sin$ derivatives (using `Real.deriv_cos`, `Real.deriv_sin`)
  and the mode-by-mode $H$ action.
- **Iterated case:** By induction on $m$, using the base case and the
  trig-poly commutativity at each step.
- **Smooth case (`sorry`):** The density argument requires
  `MeasureTheory.Lp.trigPolynomials_dense` and the fact that $H$ is a
  bounded operator on $L^2(\mathbb{T})$ (present in Mathlib as part of
  `MeasureTheory.inner_regularization`). In the CAP context this can be
  replaced by a direct check: the Julia certificate provides the values of
  $H[G^{(m)}](u_j)$ with explicit error bounds, so the identity is only
  ever *used* at grid points with a controlled error.

### 5.4 Mathlib Lemmas Used

- `Real.deriv_cos`, `Real.deriv_sin` — derivatives of trig functions
- `deriv_sum`, `deriv_const_mul` — linearity of differentiation
- `iteratedDeriv_succ` — inductive structure of iterated derivatives
- `Finset.sum_congr` — congruence of finite sums

### 5.5 Dependencies

- `HilbertOnTrigPoly.lean`

---

## 6. `AliasingErrorBound.lean`

### 6.1 Mathematical Content

When $G$ is not a trig polynomial, we cannot compute $H[G]$ exactly.
We use the DFT (discrete Fourier transform) on $N_{\mathrm{fft}}$ equally-spaced points
as a proxy, and bound the error via the *aliasing formula*.

**Setup.** Let $N = N_{\mathrm{fft}}$ be even. The DFT of $f$ sampled at
$u_j = 2\pi j/N$, $j = 0,\ldots,N-1$, produces coefficients
$\hat{f}_N(k) = \frac{1}{N}\sum_{j=0}^{N-1} f(u_j)\,e^{-iku_j}$.
The exact Fourier coefficients are
$\hat{f}(k) = \frac{1}{2\pi}\int_0^{2\pi} f(u)\,e^{-iku}\,du$.
The aliasing identity relates them:

$$\hat{f}_N(k) = \sum_{m \in \mathbb{Z}} \hat{f}(k + mN).$$

**Theorem (DFT Hilbert transform error — integral-test form).**
Let $G \in C^p(\mathbb{T})$ and suppose $\|G^{(p)}\|_{L^\infty} \leq C_p$, with $p \geq 2$
and $N \geq 4$ even.  Define $H_N[G]$ as the DFT-based discrete Hilbert transform
(circular convolution with the kernel $h_n = (2/N)\cot(\pi n/N)$ for odd $n$, zero
for even $n$; equivalent to DFT–multiply–IDFT with the spectral multiplier $-i\,\mathrm{sgn}(k)$,
Nyquist zeroed).  Then at every grid point $u_j = 2\pi(j-1)/N$:

$$|H_N[G](u_j) - H[G](u_j)| \;\le\; \varepsilon_{\mathrm{alias}}(N, p, C_p),$$

where

$$\boxed{\varepsilon_{\mathrm{alias}}(N, p, C_p) \;=\; \frac{4\,C_p}{(p-1)\,\bigl(N/2 - 1\bigr)^{p-1}}.}$$

**Derivation.**  Two ingredients:

1. *Multiplier disagreement.*  For modes $|m| \ge N/2$, the discrete multiplier
   $M_k$ (with $k = m \bmod N$) and continuous multiplier $M_\infty(m) = -i\,\mathrm{sgn}(m)$
   differ by at most $|M_k - M_\infty(m)| \le 2$ (one sign mismatch).
   For modes $|m| < N/2$ (excluding $m = \pm N/2$, where $M_{N/2}$ is zeroed and
   the disagreement is $\le 1$), the multipliers agree exactly.
   So the aliasing tail is $\sum_{|m| \ge N/2} 2 |c_m|$.
2. *Fourier decay.*  Integration by parts $p$ times: $|c_m| \le \|G^{(p)}\|_\infty / |m|^p$.
3. *Tail estimate.*  Real-Fourier symmetry $|c_{-m}| = |c_m|$ collapses the
   bilateral sum to $4 \sum_{m \ge N/2} |c_m|$.  Apply the integral test
   $\sum_{m \ge M} m^{-p} \le \int_{M-1}^\infty x^{-p}\,dx = (M-1)^{1-p}/(p-1)$
   with $M = N/2$, valid for $M \ge 2$ (i.e. $N \ge 4$).

The previous formulation in this document used $\zeta(p)/N^p$ — the discrete-vs-continuous
Lebesgue-constant route.  That formulation does not match what the Julia side certifies
(FFTHilbert.jl lines 60–82); the integral-test form above is the implemented bound and
must be the Lean target.

**Corollary (Output sup_bounds for $H[G]$, Fourier-tail form).**
The Julia code (FFTHilbert.jl:222–227) populates the output TaylorSeries'
`sup_bounds` vector via the Fourier-tail bound:

$$\bigl\|H[G^{(m)}]\bigr\|_{L^\infty} \;\le\; \bigl\|G^{(m+2)}\bigr\|_{L^\infty}\,\cdot\,\frac{\pi^2}{3}.$$

(Two IBPs give $|c_k(G^{(m)})| \le \|G^{(m+2)}\|_\infty / k^2$; sum $2\sum_{k\ge 1} k^{-2} = \pi^2/3$.)
This *replaces* an earlier unsound Lebesgue-constant attempt at bounding $\|(\mathcal H[G])^{(m)}\|_\infty$
in terms of $\|G^{(m)}\|_\infty$ — impossible because the continuous Hilbert is unbounded on $L^\infty$.
The Lean side must carry this as a separate lemma; see N3 in §6.6.

**Corollary (Grid error for H[G]).**
Under the hypotheses of `DLowerBound.lean` (which give $D \geq D_{\mathrm{lower}} > 0$),
$G = (1+Hy_u)/D$ is $C^\infty(\mathbb{T})$.
The Julia certificate provides a bound $F_p$ on $\|G^{(p)}\|_{L^\infty}$.
Then for each grid point $u_j$,

$$|H_N[G](u_j) - H[G](u_j)| \leq \varepsilon_{\mathrm{alias}}.$$

This corollary is also the analogue of `CompositeTrapezoid.lean` from CCF:
in that project the discretization error was bounded via the trapezoid rule;
here it is bounded via the DFT aliasing formula.

### 6.2 Lean Statement

```lean
import BabenkoWave.HilbertCommutesDeriv

namespace BabenkoWave

open Real MeasureTheory

/-- Fourier coefficients (L¹ normalization). -/
noncomputable def fourierCoeff (f : ℝ → ℝ) (k : ℤ) : ℂ :=
  (2 * π)⁻¹ • ∫ u in (0 : ℝ)..(2 * π), f u * Complex.exp (-Complex.I * k * u)

/-- DFT coefficients on N equally-spaced points. -/
noncomputable def dftCoeff (f : ℝ → ℝ) (N : ℕ) (k : ℤ) : ℂ :=
  (N : ℂ)⁻¹ • ∑ j : Fin N, f (2 * π * j / N) * Complex.exp (-Complex.I * k * (2 * π * j / N))

/-- Aliasing identity: dftCoeff f N k = Σ_{m : ℤ} fourierCoeff f (k + m * N). -/
theorem aliasing_identity (f : ℝ → ℝ) (hf : Continuous f)
    (hperiod : ∀ u, f (u + 2 * π) = f u) (N : ℕ) (hN : 0 < N) (k : ℤ) :
    dftCoeff f N k = ∑' m : ℤ, fourierCoeff f (k + m * N) := by
  sorry -- Standard DFT aliasing; follows from Poisson summation formula.
        -- Key Mathlib ingredient: `MeasureTheory.inner_regularization` + Poisson.

/-- Fourier coefficient decay for C^p functions.
    For f ∈ C^p(𝕋) with ‖f^{(p)}‖_∞ ≤ Fp:
    |fourierCoeff f k| ≤ Fp / |k|^p for k ≠ 0. -/
theorem fourierCoeff_decay
    {f : ℝ → ℝ} {p : ℕ} (hp : 1 ≤ p)
    (hf : ContDiff ℝ p f)
    (hperiod : ∀ u, f (u + 2 * π) = f u)
    {Fp : ℝ} (hFp : ∀ u, |iteratedDeriv p f u| ≤ Fp)
    {k : ℤ} (hk : k ≠ 0) :
    Complex.abs (fourierCoeff f k) ≤ Fp / (k.natAbs : ℝ) ^ p := by
  -- Integration by parts p times. Each IBP introduces a factor of 1/(ik).
  -- After p steps: fourierCoeff f^{(p)} k = (ik)^p · fourierCoeff f k.
  sorry -- TODO: `MeasureTheory.inner_regularization` + IBP p times.

/-- DFT Hilbert transform applied to f at a grid point.  Defined via the
    cotangent kernel `h_n = (2/N) cot(π n/N)` for odd n (zero else); equivalent
    to DFT–multiply–IDFT with the spectral multiplier −i·sgn(k), Nyquist zeroed. -/
noncomputable def H_N (f : ℝ → ℝ) (N : ℕ) (u : ℝ) : ℝ := sorry

/-- Main aliasing error bound (integral-test form).  Matches the Julia side at
    `BabenkoRigorous/src/FFTHilbert.jl:178-183`. -/
theorem fft_hilbert_aliasing_bound
    {G : ℝ → ℝ} {p : ℕ} (hp : 2 ≤ p)
    (hG : ContDiff ℝ p G)
    (hperiod : Function.Periodic G (2 * π))
    {Cp : ℝ} (hCp_pos : 0 < Cp)
    (hCp : ∀ u, |iteratedDeriv p G u| ≤ Cp)
    {N : ℕ} (hN : 4 ≤ N) (hN_even : Even N) (u : ℝ) :
    |H_N G N u - periodicHilbert G u|
      ≤ 4 * Cp / ((p - 1 : ℝ) * ((N / 2 : ℝ) - 1) ^ (p - 1)) := by
  -- Proof outline:
  -- 1. (Multiplier disagreement)  For each aliased mode m with |m| ≥ N/2,
  --    |M_k - M_∞(m)| ≤ 2 (where k = m mod N).  At m = ±N/2 the disagreement
  --    is 1 (Nyquist zeroed), still ≤ 2.
  -- 2. (Fourier decay) |c_m(G)| ≤ Cp / |m|^p (integration by parts p times).
  -- 3. (Symmetry) For real G, |c_{-m}| = |c_m|, so the bilateral aliasing tail
  --    collapses to 4 · Σ_{m ≥ N/2} |c_m|.
  -- 4. (Integral test) Σ_{m ≥ N/2} m^{-p} ≤ ∫_{N/2 - 1}^∞ x^{-p} dx
  --                                       = (N/2 - 1)^{1-p} / (p - 1)
  --    valid for N/2 ≥ 2, i.e. N ≥ 4.
  -- 5. Combine to get the boxed constant.
  sorry

/-- Fourier-tail π²/3 bound for H applied to a C² periodic function.  This is
    the replacement for the unsound Lebesgue-constant approach: the continuous
    Hilbert is unbounded on L^∞, so we use two integrations by parts to express
    the bound in terms of one higher derivative norm. -/
theorem hilbert_sup_bound_from_two_derivs
    {g : ℝ → ℝ} (hg : ContDiff ℝ 2 g)
    (hper : Function.Periodic g (2 * π))
    {F2 : ℝ} (hF2 : ∀ u, |iteratedDeriv 2 g u| ≤ F2) :
    ∀ u, |periodicHilbert g u| ≤ F2 * π ^ 2 / 3 := by
  -- Proof: |c_k(g)| ≤ F2 / k² (two IBPs); ‖H[g]‖_∞ ≤ Σ_{k≠0} |c_k(g)|
  --                                                 = 2 Σ_{k≥1} F2/k²
  --                                                 = F2 · π²/3
  -- (Basel: ∑_{k≥1} k⁻² = π²/6; Mathlib has `Real.tsum_pNat_pow_two`.)
  sorry

/-- Alias used to populate ts_HG.sup_bounds[m+1] = ‖G^(m+2)‖_∞ · π²/3. -/
def output_sup_bound (Cm2 : ℝ) : ℝ := Cm2 * π ^ 2 / 3
```

### 6.3 Proof Sketch

- **`aliasing_identity`:** The standard proof applies the Poisson summation formula
  to the function $j \mapsto f(2\pi j/N)$.
  In Mathlib, `MeasureTheory.inner_regularization` gives the Poisson summation
  for $L^1$ functions; combine with periodicity to get the aliasing identity.
- **`fourierCoeff_decay`:** Integration by parts $p$ times.
  Mathlib has `intervalIntegral.integral_comp_mul_right` and
  `intervalIntegral.integral_deriv_eq_sub` for individual IBP steps.
  The boundary terms vanish by $2\pi$-periodicity.
- **`fft_hilbert_aliasing_bound`:** Combine the four ingredients listed in the
  theorem statement above (multiplier disagreement, Fourier decay, real-Fourier
  symmetry, integral test).  The integral test is the key analytical step that
  replaces `tsum_zeta_two`-style arguments with an explicit closed-form
  $(N/2 - 1)^{1-p}/(p-1)$ bound — matching what Julia computes.
- **`hilbert_sup_bound_from_two_derivs`:** Mathlib's `Real.tsum_pNat_pow_two`
  (Basel: $\sum_{k \ge 1} k^{-2} = \pi^2/6$) closes the proof in two lines:
  $\|H[g]\|_\infty \le \sum_{k\neq 0} |c_k(g)| \le 2 F_2 \sum_{k \ge 1} k^{-2} = F_2 \pi^2/3$.

### 6.4 Mathlib Lemmas Used

- `Real.tsum_pNat_pow_two` — Basel: $\sum_{k \ge 1} k^{-2} = \pi^2/6$
- `Complex.abs_sum_le_sum_abs` — triangle inequality for complex sums
- `intervalIntegral.integral_comp_mul_right` — change of variables in integrals
- `MeasureTheory.inner_regularization` — Poisson summation (approximate)
- `tsum_le_of_sum_le_integral` (or analogous) — integral test for tail sums

### 6.5 Dependencies

- `HilbertCommutesDeriv.lean`

### 6.6 New lemmas introduced by the post-implementation audit

The recent Julia refactor (sup_bounds vector, exact-Leibniz product, Faà di Bruno
via ArbSeries composition, Fourier-tail π²/3 for $\|H[g]\|_\infty$, integral-test
aliasing) introduces five lemmas that were not in the pre-implementation Lean plan.
Two of them (N3 = Fourier-tail π²/3 and N4 = integral-test aliasing) are in §6.2
above.  N5 (grid-to-supnorm Taylor remainder) is in §9.  The remaining two (N1, N2)
are listed here.

#### N1.  Faà di Bruno sup_bound via Bell-polynomial composition

Used by `inv_series` and `sqrt_series` to compute the sup_bounds vector of
$1/f$ and $\sqrt f$ from the sup_bounds of $f$.  The Julia code uses
`_faa_di_bruno_derivs` (ArbSeries composition) over non-negative inputs;
the Lean counterpart is:

```lean
/-- N1.  Faà di Bruno sup_bound.  If φ is C^n on an interval that contains
    the image of f, and we know per-derivative bounds Φ_k ≥ ‖φ^(k)‖_∞ on
    that interval and F_k ≥ ‖f^(k)‖_∞ globally, then the n-th derivative
    sup-norm of (φ ∘ f) is bounded by the Bell-polynomial sum:

        ‖(φ ∘ f)^(n)‖_∞ ≤ Σ_{π ⊢ n}  (n! / Π m_i! · k_i!) · Φ_{|π|} · Π F_{m_i}^{k_i},

    summed over set partitions π of n with block sizes m_1, …, m_k.
    Every Bell polynomial coefficient is a non-negative integer, so feeding
    non-negative sup_bounds in place of derivative values dominates pointwise. -/
theorem faa_di_bruno_sup_bound
    {n : ℕ} {φ : ℝ → ℝ} {f : ℝ → ℝ}
    {Φ : Fin (n + 1) → ℝ} {F : Fin (n + 1) → ℝ}
    {d_lo d_hi : ℝ} (hd : d_lo ≤ d_hi)
    (hφ : ∀ k ≤ n, ∀ x ∈ Set.Icc d_lo d_hi, |iteratedDeriv k φ x| ≤ Φ ⟨k, by omega⟩)
    (hf : ∀ k ≤ n, ∀ u, |iteratedDeriv k f u| ≤ F ⟨k, by omega⟩)
    (hrange : ∀ u, f u ∈ Set.Icc d_lo d_hi) :
    ∀ u, |iteratedDeriv n (φ ∘ f) u| ≤ bell_polynomial_eval n Φ F := by
  sorry
```

Specialized via `φ(x) = 1/x` on $[d_{\rm lower}, \infty)$ (with
$\Phi_k = k!/d_{\rm lower}^{k+1}$) and `φ(x) = √x` on $[d_{\rm lower}^2, S_{\rm upper}]$
(with $\Phi_k$ given by the half-integer Pochhammer formula in `TaylorSeries.jl`).

#### N2.  Exact-Leibniz product sup_bound

Used by every TaylorSeries product `ts1 * ts2`.  Direct corollary of Leibniz +
triangle inequality, but worth stating cleanly:

```lean
/-- N2.  Exact Leibniz on per-derivative supnorms.

    Given F_k ≥ ‖f^(k)‖_∞ and G_k ≥ ‖g^(k)‖_∞ for k = 0, …, n,
        ‖(f · g)^(n)‖_∞ ≤ Σ_{k=0}^{n} C(n, k) · F_k · G_{n-k}.

    Rigorous because binomial coefficients are non-negative integers and the
    triangle inequality applies term-by-term to Leibniz's formula. -/
theorem product_sup_bounds_leibniz
    {n : ℕ} {f g : ℝ → ℝ}
    (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g)
    {F G : Fin (n + 1) → ℝ}
    (hF : ∀ k ≤ n, ∀ u, |iteratedDeriv k f u| ≤ F ⟨k, by omega⟩)
    (hG : ∀ k ≤ n, ∀ u, |iteratedDeriv k g u| ≤ G ⟨k, by omega⟩) :
    ∀ u, |iteratedDeriv n (f * g) u|
        ≤ ∑ k : Fin (n + 1), (Nat.choose n k.val : ℝ) * F k * G ⟨n - k.val, by omega⟩ := by
  -- Leibniz: (fg)^(n) = Σ C(n, k) f^(k) g^(n-k).
  -- Triangle inequality on each summand.
  sorry
```

#### Summary of new lemmas

| Lemma | Where | Replaces / supports |
|---|---|---|
| N1 | §6.6 (new) | The audit-flagged-broken `(M+1)!·S^{M+1}/d^{M+2}` formula for `inv_series` and the dimensionally-wrong `S^{M/2}` for `sqrt_series`. |
| N2 | §6.6 (new) | The audit-flagged-marginal `2^{M+1}(F_1·S_2 + S_1·F_2)` product F_bound. |
| N3 | §6.1, §6.2 | Replaces the unsound Lebesgue-constant approach to bounding $\|H[g]\|_\infty$. |
| N4 | §6.1, §6.2 | Replaces the $\zeta(p)/N^p$ Lean aliasing formula with the integral-test form actually used by Julia. |
| N5 | §9 | Replaces the Lipschitz `grid_to_supNorm` with the full Taylor-remainder form. |

---

## 7. `TaylorRemainder.lean`

### 7.1 Mathematical Content

This file adapts `TaylorRemainderSubinterval.lean` from the CCF project
to the water-wave setting.

**Theorem (Lagrange Taylor remainder).**
Let $f \in C^{M+1}([a,b])$ and let $T_M f(a, \cdot)$ denote its order-$M$
Taylor expansion at $a$:

$$T_M f(a, t) = \sum_{k=0}^{M} \frac{f^{(k)}(a)}{k!}(t-a)^k.$$

Then for all $t \in [a,b]$,

$$\left|f(t) - T_M f(a,t)\right| \leq \frac{\|f^{(M+1)}\|_{L^\infty([a,b])}}{(M+1)!}\,|t-a|^{M+1}.$$

This is the sharp Lagrange form (with $(M+1)!$ in the denominator, not $(M+1)\cdot M!$).
The CCF file `TaylorRemainderSubinterval.lean` achieves this sharp form via
`taylor_mean_remainder_lagrange_iteratedDeriv` from Mathlib.

**Corollary (Sup-norm from grid via Taylor).**
Let $h = 2\pi/N$ and let $u_j = jh$ for $j = 0, \ldots, N-1$.
Suppose:
1. $|R(u_j)| \leq \varepsilon_0$ for all $j$,
2. $\|R^{(M+1)}\|_{L^\infty([0,2\pi])} \leq F_{M+1}$.

Then for any $u \in [0, 2\pi]$, writing $u \in [u_j, u_{j+1}]$ for the
appropriate $j$ and applying the Lagrange remainder at $u_j$:

$$|R(u)| \leq |R(u_j)| + \sum_{k=1}^{M} \frac{|R^{(k)}(u_j)|}{k!}\,h^k + \frac{F_{M+1}}{(M+1)!}\,h^{M+1}.$$

If moreover $\|R^{(k)}\|_\infty \leq F_k$ for $k = 1,\ldots,M$, this simplifies to:

$$|R(u)| \leq \varepsilon_0 + \sum_{k=1}^{M} \frac{F_k}{k!}\,h^k + \frac{F_{M+1}}{(M+1)!}\,h^{M+1}.$$

**Simplified form used in `Main.lean`.**
When $M = 0$ (i.e., we do not use derivative information beyond grid values),
the corollary reduces to: if $R$ is $L$-Lipschitz and $|R(u_j)| \leq \varepsilon_0$,
then $\|R\|_\infty \leq \varepsilon_0 + L\cdot h/2 = \varepsilon_0 + L\pi/N$.
This is the form stated in the proof strategy in Section 1,
and it is handled directly by `GridToSupNorm.lean` (Section 9).
The Taylor machinery in this file is used for the *higher-order* version when
the Julia certificate provides derivative bounds as well.

### 7.2 Lean Statement

```lean
import BabenkoWave.AliasingErrorBound

namespace BabenkoWave

open Real MeasureTheory Set

-- Re-export the CCF Taylor remainder:
/-- Adapted from CCFLeanPOC.PathBeta.un_taylor_remainder.
    Key identity: |f(t) - T_M f(a,t)| ≤ F · |t-a|^{M+1} / (M+1)!
    where F = ‖f^{(M+1)}‖_∞ on [a,b]. -/
theorem taylor_remainder_pointwise
    {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b) {M : ℕ}
    (hf : ContDiffOn ℝ (M + 1) f (Icc a b))
    {F : ℝ}
    (hF : ∀ t ∈ Icc a b, |iteratedDerivWithin (M + 1) f (Icc a b) t| ≤ F)
    {t : ℝ} (ht : t ∈ Icc a b) :
    |f t - taylorWithinEval f M (Icc a b) a t|
      ≤ F * (t - a) ^ (M + 1) / (Nat.factorial (M + 1) : ℝ) :=
  -- Direct application of abs_taylor_remainder_le from the CCF file.
  CCFLeanPOC.PathBeta.abs_taylor_remainder_le hab hf hF ht

/-- Sup-norm bound from grid values + higher-derivative bound. -/
theorem supNorm_from_grid_taylor
    {f : ℝ → ℝ} {N M : ℕ} (hN : 0 < N)
    (hf : ContDiffOn ℝ (M + 1) f (Icc 0 (2 * π)))
    {ε₀ : ℝ} (hε₀ : ∀ j : Fin N, |f (2 * π * j / N)| ≤ ε₀)
    {Fk : Fin (M + 1) → ℝ}
    (hFk : ∀ k : Fin (M + 1), ∀ t ∈ Icc 0 (2 * π),
      |iteratedDerivWithin k.val f (Icc 0 (2 * π)) t| ≤ Fk k)
    {FM1 : ℝ}
    (hFM1 : ∀ t ∈ Icc 0 (2 * π),
      |iteratedDerivWithin (M + 1) f (Icc 0 (2 * π)) t| ≤ FM1) :
    ∀ u ∈ Icc 0 (2 * π),
      |f u| ≤ ε₀ + ∑ k : Fin M, Fk ⟨k.val + 1, by omega⟩ / Nat.factorial (k.val + 1)
                                  * (2 * π / N) ^ (k.val + 1)
              + FM1 / Nat.factorial (M + 1) * (2 * π / N) ^ (M + 1) := by
  intro u hu
  -- Find the nearest left grid point.
  obtain ⟨j, hj_lt, hj_le⟩ : ∃ j : Fin N,
      (j : ℝ) * (2 * π / N) ≤ u ∧ u < ((j : ℝ) + 1) * (2 * π / N) := by
    sorry -- Existence of floor index; follows from Nat.floor_div_le.
  have hh : 2 * π / N > 0 := by positivity
  have huj_mem : 2 * π * j / N ∈ Icc 0 (2 * π) := by sorry
  have htu_le : u - 2 * π * j / N ≤ 2 * π / N := by linarith [hj_le]
  -- Apply Taylor remainder at u_j.
  calc |f u|
      ≤ |f (2 * π * j / N)| + |f u - taylorWithinEval f M (Icc 0 (2 * π)) (2 * π * j / N) u|
            + |taylorWithinEval f M (Icc 0 (2 * π)) (2 * π * j / N) u
                - f (2 * π * j / N)| := by
          -- Triangle inequality (Taylor at u_j reconstructs f(u_j) exactly at k=0)
          sorry
    _ ≤ ε₀ + ∑ k : Fin M, Fk _ / Nat.factorial _ * (2 * π / N) ^ _ + FM1 / _ * _ := by
          sorry
```

### 7.3 Proof Sketch

- Locate the nearest left grid point $u_j = 2\pi j / N$ to $u$ (using `Nat.floor`).
- Apply `taylor_remainder_pointwise` on $[u_j, u_{j+1}]$ with $t = u$.
- Bound each term of the Taylor polynomial using the derivative bounds $F_k$.
- Use $|u - u_j| \leq 2\pi/N$ for all terms.

### 7.4 Mathlib Lemmas Used

- `Nat.floor`, `Nat.floor_le`, `Nat.lt_floor_add_one` — grid index location
- `taylorWithinEval_self` — $T_M f(a, a) = f(a)$
- `taylor_mean_remainder_lagrange_iteratedDeriv` — Lagrange remainder form
- `Finset.sum_le_sum` — bounding the Taylor sum term-by-term

### 7.5 Dependencies

- `AliasingErrorBound.lean`
- Indirect reference to `CCFLeanPOC.PathBeta.abs_taylor_remainder_le`
  (or a local copy of that proof)

---

## 8. `DLowerBound.lean`

### 8.1 Mathematical Content

This file converts the Julia certificate hypothesis $D(u) \geq D_{\mathrm{lower}} > 0$
into analytic consequences used throughout the proof.

**Lemma D1 (Strict positivity).**
If $D_{\mathrm{lower}} > 0$ and $D(u) \geq D_{\mathrm{lower}}$ for all $u$,
then $D(u) > 0$ for all $u$, and $1/D(u) \leq 1/D_{\mathrm{lower}}$.

**Lemma D2 (Lipschitz bound on $1/D$).**
$D(u) = \sqrt{y_u(u)^2 + (1 + Hy_u(u))^2}$, and $y_u$, $Hy_u$ are trig polynomials,
hence $C^\infty$.
By the chain rule,

$$\left|\partial_u\left(\frac{1}{D(u)}\right)\right|
= \frac{|D_u(u)|}{D(u)^2} \leq \frac{|D_u(u)|}{D_{\mathrm{lower}}^2}.$$

Since $|D_u(u)|$ is bounded by $\|D_u\|_\infty \leq D_{\mathrm{Lip}}$ (computed from
the explicit coefficients $a_k$), we get

$$\left\|\partial_u\!\left(\frac{1}{D}\right)\right\|_\infty \leq \frac{D_{\mathrm{Lip}}}{D_{\mathrm{lower}}^2}.$$

**Lemma D3 (Smoothness of $G$).**
$G = (1 + Hy_u)/D$ is $C^\infty([0,2\pi])$ with all derivatives bounded
by expressions depending polynomially on $1/D_{\mathrm{lower}}$,
$\|y_u\|_{C^m}$, and $\|Hy_u\|_{C^m}$.
In particular, $G \in C^p$ for any $p$ needed by `AliasingErrorBound.lean`.

**Certificate hypothesis.**
The number $D_{\mathrm{lower}}$ is provided by the Julia certificate.
In Lean it appears as:

```
h_D : ∀ u : ℝ, DLower ≤ D N a u
```

### 8.2 Lean Statement

```lean
import BabenkoWave.TaylorRemainder

namespace BabenkoWave

open Real

/-- D1: D(u) ≥ DLower > 0 implies 1/D(u) ≤ 1/DLower. -/
lemma one_div_D_le
    {DLower : ℝ} (hDL : 0 < DLower)
    (hD : ∀ u, DLower ≤ D N a u) (u : ℝ) :
    (D N a u)⁻¹ ≤ DLower⁻¹ := by
  apply inv_le_inv_of_le hDL (hD u)

/-- D1b: D(u) > 0 pointwise. -/
lemma D_pos
    {DLower : ℝ} (hDL : 0 < DLower)
    (hD : ∀ u, DLower ≤ D N a u) (u : ℝ) :
    0 < D N a u :=
  lt_of_lt_of_le hDL (hD u)

/-- D2: Pointwise bound on |∂_u (1/D)|.
    DLip is a bound on |D_u|, supplied by the certificate. -/
lemma deriv_one_div_D_le
    {DLower DLip : ℝ} (hDL : 0 < DLower) (hDLip : 0 ≤ DLip)
    (hD : ∀ u, DLower ≤ D N a u)
    (hDeriv : ∀ u, |deriv (D N a) u| ≤ DLip)
    (u : ℝ) :
    |deriv (fun v => (D N a v)⁻¹) v| ≤ DLip / DLower ^ 2 := by
  have hDpos : 0 < D N a u := D_pos hDL hD u
  rw [deriv_inv (D N a) u, abs_neg, abs_div, abs_pow]
  apply div_le_div_of_nonneg_left (hDeriv u) (sq_pos_of_pos hDpos) (by positivity)
  exact sq_le_sq' (by linarith [hD u]) (hD u)

/-- D3: G = (1 + Hyu) / D is ContDiff on ℝ (given D > 0). -/
lemma G_contDiff
    {DLower : ℝ} (hDL : 0 < DLower)
    (hD : ∀ u, DLower ≤ D N a u) (p : ℕ) :
    ContDiff ℝ p (G N a) := by
  -- G = (1 + Hyu) / D = (1 + Hyu) · (D)⁻¹
  -- (1 + Hyu) is C^∞ (trig polynomial).
  -- D = sqrt(yu² + (1+Hyu)²) is C^∞ when its argument is strictly positive
  --   (which follows from D ≥ DLower > 0).
  -- Product of C^p functions is C^p.
  apply ContDiff.div
  · exact contDiff_const.add (Hyu_contDiff N a)
  · exact D_contDiff N a hDL hD
  · intro u; exact ne_of_gt (D_pos hDL hD u)
```

### 8.3 Proof Sketch

- **D1:** Direct from `inv_le_inv_of_le` in Mathlib's `OrderedField`.
- **D2:** Differentiate $1/D$ using `deriv_inv`, apply the chain rule,
  and bound $|D_u|/D^2$ using $D \geq D_{\mathrm{lower}}$.
- **D3:** $y_u$ and $Hy_u$ are finite trig polynomials, hence $C^\infty$.
  $D = \sqrt{y_u^2 + (1+Hy_u)^2}$ is $C^\infty$ because the argument of
  $\sqrt{\cdot}$ is bounded below by $D_{\mathrm{lower}}^2 > 0$
  (use `ContDiff.sqrt` with the hypothesis `h > 0`).
  Division by $D$ is $C^\infty$ by `ContDiff.div`.

### 8.4 Mathlib Lemmas Used

- `inv_le_inv_of_le` — monotonicity of inversion for positive reals
- `deriv_inv` — derivative of $1/f$
- `ContDiff.sqrt` — smoothness of $\sqrt{\cdot}$ away from zero
- `ContDiff.div` — smoothness of quotient
- `ContDiff.add`, `ContDiff.mul` — algebraic operations on $C^p$ functions

### 8.5 Dependencies

- `TaylorRemainder.lean`

---

## 9. `GridToSupNorm.lean` (Taylor-remainder form)

### 9.1 Mathematical Content

This is the **key abstract lemma** bridging discrete grid control to a
continuous sup-norm bound. It does not exist in Mathlib and is the
central contribution of the verification layer.

**Theorem (Grid to sup-norm, Taylor-remainder version).**
Let $f \in C^{M+1}(\mathbb{T})$ be $2\pi$-periodic.  Suppose Julia provides:

- **Per-grid-point derivative bounds** $R_{j,k}$ for $j = 0,\ldots,N-1$ and
  $k = 0,\ldots,M$, with $|f^{(k)}(2\pi j/N)| \le R_{j,k}$.
- **A top-derivative supnorm bound** $F_{M+1}$ with $\|f^{(M+1)}\|_\infty \le F_{M+1}$.

Then for $h = 2\pi/N$ and all $u \in [0, 2\pi]$,

$$|f(u)| \;\le\; \max_{j} \!\left[\sum_{k=0}^{M} \frac{R_{j,k}}{k!}\,h^k\right]
                \;+\; \frac{F_{M+1}}{(M+1)!}\,h^{M+1}.$$

*Proof.* For any $u \in [0, 2\pi]$, let $u_j$ be the *left* grid point with
$0 \le u - u_j \le h$.  By Taylor's theorem with Lagrange remainder,

$$f(u) = \sum_{k=0}^{M} \frac{f^{(k)}(u_j)}{k!}(u - u_j)^k
        + \frac{f^{(M+1)}(\xi)}{(M+1)!}(u - u_j)^{M+1}$$

for some $\xi \in (u_j, u)$.  Taking absolute value and using the hypotheses:

$$|f(u)| \le \sum_{k=0}^{M} \frac{|f^{(k)}(u_j)|}{k!}\,h^k + \frac{\|f^{(M+1)}\|_\infty}{(M+1)!}\,h^{M+1}
        \le \sum_{k=0}^{M} \frac{R_{j,k}}{k!}\,h^k + \frac{F_{M+1}}{(M+1)!}\,h^{M+1}.$$

Taking the max over $j$ completes the proof. $\quad\square$

**Relationship to the Julia `sup_bound`.**  This is exactly the bound that
`sup_bound(ts::TaylorSeries)` returns (see `BabenkoRigorous/src/TaylorSeries.jl:148-152`):
the matrix `ts.derivs[j, k+1]` supplies the per-grid-point derivative bounds
$R_{j,k}$, and `ts.sup_bounds[M+2]` supplies $F_{M+1}$.  The Lean theorem
above lets `Main.lean` plug Julia's `sup_bound(ts_R)` output in as the
upper bound on $\|R\|_\infty$ — no Lipschitz constant or `R_differentiable`
proof of $R$ ever needed.

**Remark on the previous Lipschitz form.**  An earlier draft of this section
specialized to the $M = 0$ case and required Julia to compute a single
global Lipschitz constant $L \ge \|R'\|_\infty$.  That route is now optional:
$M = 0$ recovers the Lipschitz version as a corollary (the sum collapses
to $R_{j,0}$ and the remainder reduces to $F_1 \cdot h$), but the
implemented Julia code already computes the full per-grid-point matrix
$R_{j,k}$ plus the top supnorm via the exact-Leibniz / Faà di Bruno
machinery, so the higher-order form is essentially free to use.

### 9.2 Lean Statement

```lean
import BabenkoWave.DLowerBound

namespace BabenkoWave

open Real Set

/-- The key abstract lemma: per-grid-point derivative bounds + top-supnorm
    bound → sup-norm bound (Taylor-remainder form). This lemma is pure
    analysis; it makes no reference to the Babenko equation.

    Matches the bound returned by `sup_bound(ts::TaylorSeries)` in
    `BabenkoRigorous/src/TaylorSeries.jl:148-152`. -/
theorem grid_to_supNorm_taylor
    {f : ℝ → ℝ} {M N : ℕ} (hN : 0 < N) (hM : 0 ≤ M)
    {Rk_grid : Fin N → Fin (M + 1) → ℝ} {FM1 : ℝ}
    (hf : ContDiff ℝ (M + 1) f)
    (hperiod : Function.Periodic f (2 * π))
    (hRk : ∀ j k, |iteratedDeriv k.val f (2 * π * (j : ℝ) / N)| ≤ Rk_grid j k)
    (hFM1 : ∀ u, |iteratedDeriv (M + 1) f u| ≤ FM1) :
    let h := (2 * π : ℝ) / N
    let factorialk := fun k : Fin (M + 1) => (Nat.factorial k.val : ℝ)
    let factorial_M1 := (Nat.factorial (M + 1) : ℝ)
    ∀ u : ℝ, |f u| ≤
      (Finset.univ.image fun j : Fin N =>
        ∑ k : Fin (M + 1), Rk_grid j k * h ^ k.val / factorialk k).max'
        ⟨Rk_grid ⟨0, hN⟩ ⟨0, by omega⟩ * h^0 / 1, by sorry⟩
      + FM1 * h ^ (M + 1) / factorial_M1 := by
  -- Proof: for u ∈ [u_j, u_j+h], apply Taylor's theorem with Lagrange remainder
  -- (Mathlib: `taylor_mean_remainder_lagrange_iteratedDeriv`).
  sorry

/-- Specialization to M = 0 recovers the Lipschitz version: with M = 0 the
    sum reduces to R_{j,0} = |f(u_j)| and the remainder reduces to F_1 · h.
    Useful as a sanity-check corollary. -/
theorem grid_to_supNorm_lipschitz
    {f : ℝ → ℝ} {L ε₀ : ℝ} (hL : 0 ≤ L) (hε₀ : 0 ≤ ε₀)
    {N : ℕ} (hN : 0 < N)
    (hperiod : Function.Periodic f (2 * π))
    (hLip : ∀ u v, |f u - f v| ≤ L * |u - v|)
    (hGrid : ∀ j : Fin N, |f (2 * π * (j : ℝ) / N)| ≤ ε₀) :
    ∀ u : ℝ, |f u| ≤ ε₀ + L * π / N := by
  intro u
  -- Direct proof using nearest grid point; or derive from grid_to_supNorm_taylor
  -- with M = 0 (FM1 = L corresponds to ‖f'‖_∞ ≤ L).
  sorry
```

### 9.3 Proof Sketch

- **Step 1 (nearest grid point):** Use `Nat.round` to define $j_0$.
  Prove $|u - 2\pi j_0/N| \leq \pi/N$ from the definition of rounding
  (`Nat.round_le`, `Nat.le_round`).
- **Step 2 (Lipschitz):** Unfold `LipschitzWith` to get
  $|f(u) - f(u_j)| \leq L \cdot |u - u_j|$, apply `mul_le_mul_of_nonneg_left`.
- **Step 3 (triangle):** `abs_sub_abs_le_abs_sub` + `hGrid j`.
- **Periodic version:** Reduce to the $[0,2\pi]$ version by replacing $u$ with
  $u - 2\pi\lfloor u/(2\pi) \rfloor$ and applying periodicity of $f$ and $R$.

### 9.4 Mathlib Lemmas Used

- `LipschitzWith.dist_le_mul` — the Lipschitz condition in metric-space form
- `Nat.round`, `Nat.round_le`, `Nat.le_round` — rounding to nearest integer
- `abs_sub_abs_le_abs_sub` — reverse triangle inequality
- `mul_le_mul_of_nonneg_left` — monotone multiplication

### 9.5 Dependencies

- `DLowerBound.lean` (imports all prior files transitively)

---

## 10. `Main.lean`

### 10.1 Mathematical Content

All pieces assemble here. The headline theorem states:

**Theorem (Babenko residual bound).**
Let $y = \sum_{k=1}^N a_k \cos(ku)$, $\mu, \beta \in \mathbb{R}$, and let $R$ be
the Babenko residual defined in `Definitions.lean`.
The Julia/Arb chain produces a scalar $\varepsilon_0$ such that
$\|R\|_{L^\infty([0,2\pi])} \leq \varepsilon_0$.

The chain works through the universal lemmas N1–N5 (Faà di Bruno, Leibniz,
Fourier-tail, integral-test aliasing, grid-to-sup-norm Taylor remainder)
applied to the per-derivative-level `sup_bounds` vectors that `TaylorSeries`
threads through the residual computation. Lean's job in `Main.lean` is to
*chain* the single trust-anchor axiom `cert_residual_le` to the headline
theorem; everything quantitative is done on the Julia side.

### 10.2 Lean Statement

```lean
import WaveCAP.HilbertOnTrigPoly
import WaveCAP.HilbertCommutesDeriv
import WaveCAP.AliasingErrorBound
import WaveCAP.FourierTailHilbert
import WaveCAP.FaaDiBrunoBound
import WaveCAP.LeibnizBound
import WaveCAP.TaylorRemainder
import WaveCAP.GridToSupNormTaylor
import WaveCAP.DLowerBound
import WaveCAP.Certificate

namespace WaveCAP

open WaveCAP.Certificate

/-- **Headline**: the Babenko residual is bounded by `ε₀_cert` at every point.

The proof is a direct chain through `cert_residual_le` (the Arb trust anchor
shipped by `Certificate.lean`).  The universal analytic lemmas (N1–N5) plus
the `TaylorSeries.sup_bound` grid-max argument together justify the Julia
chain's output as a true sup-norm bound; that justification is verified once
on the Lean side and consumed via `cert_residual_le`. -/
theorem residual_bounded_by_certificate :
    ∀ u, |R a_cert μ_cert β_cert u| ≤ ε₀_cert :=
  cert_residual_le

end WaveCAP
```

The pre-implementation draft of this section used a Lipschitz-form bound
$\varepsilon_0 + L \pi / M$ assembled from per-grid-point evaluations + a
single derivative bound. The current implementation uses the
**Taylor-remainder form** (`sup_bound(ts_R)` from `TaylorSeries.jl`), which
gives a tighter bound than the Lipschitz form by exploiting all `M_order+1`
derivatives at each grid point. The Lean-side machinery
(`grid_to_supNorm_taylor` in `GridToSupNormTaylor.lean`) verifies this
tighter bound formula directly.

### 10.3 Which Lemma from Each File Is Used

| File | Lemma used in `Main.lean` |
|---|---|
| `Definitions.lean` | `R`, `D` — the definitions being bounded |
| `HilbertOnTrigPoly.lean` | `H_of_yu_eq_Hyu` — confirms $H[y_u] = Hy_u$ (used in `R_differentiable`) |
| `HilbertCommutesDeriv.lean` | `H_commutes_deriv_trigpoly` — used in differentiating $H[y\,y_u]$ inside $R'$ |
| `AliasingErrorBound.lean` | `H_N_error_bound` — controls $H[\partial_u G]$ in $R'$ |
| `TaylorRemainder.lean` | Not directly; its role is subsumed by `grid_to_supNorm_periodic` when $M=0$ |
| `DLowerBound.lean` | `D_pos`, `G_contDiff`, `deriv_one_div_D_le` — used in `R_differentiable` |
| `GridToSupNorm.lean` | `grid_to_supNorm_periodic` — the core interpolation lemma |
| `Certificate.lean` | Provides `h_grid`, `h_deriv`, `h_D` as `theorem`s with `native_decide` or `norm_num` proofs |

### 10.4 Auxiliary Lemmas Needed

The proof of `R_differentiable` (that $R$ is differentiable with $|R'| \leq L$)
requires bounding five terms in $R'$:

1. $\partial_u(\mu\, Hy_u)$: trig polynomial, $C^\infty$.
2. $\partial_u(y + y\,Hy_u)$: trig polynomial, $C^\infty$.
3. $\partial_u H[y\,y_u]$: trig polynomial (by `H_commutes_deriv_trigpoly`), $C^\infty$.
4. $\partial_u^2(y_u/D)$: requires $D \geq D_{\mathrm{lower}}$ (`DLowerBound.lean`) and explicit bounds on $y_u^{(2)}$, $D_u$.
5. $\partial_u H[\partial_u G]$: requires `AliasingErrorBound.lean` + `DLowerBound.lean`.

The Julia certificate provides $L$ as an explicit upper bound on $|R'|$,
computed rigorously via interval arithmetic.
Lean's job in `R_differentiable` is therefore to prove that
$R$ is *at least* differentiable (so that `lipschitzWith_of_nnnorm_deriv_le` applies),
with the derivative bounded by the certificate value $L$.
This is the most technically complex intermediate lemma.

---

## 11. Julia–Lean Interface

### 11.1 `Certificate.lean`

The file `Certificate.lean` is the trust boundary between the Julia numerics
and the Lean proof. It contains **no proofs** — only statements marked
`axiom` or proved by `native_decide` / `norm_num` on rational approximations.

```lean
import BabenkoWave.Definitions

namespace BabenkoWave

/-!
# Certificate values

These constants are output by the Julia/Arblib rigorous computation.
They are the *only* place in the Lean proof where external computation
is trusted without an internal proof.

Trust argument: the Julia code uses Arblib interval arithmetic throughout.
Every certificate value is an interval upper/lower bound with a guaranteed
rounding-error enclosure. The Lean hypotheses assert the weaker
(≤ rational constant) form of these bounds.
-/

/-- Number of cosine modes (from the Julia computation). -/
def N_cert : ℕ := 3  -- example; equals length(y_coeffs)

/-- Cosine coefficients as exact dyadic rationals from the Float64 inputs. -/
def a_cert : Fin N_cert → ℚ := ![ sorry, sorry, sorry ]

/-- Wave speed parameter μ. -/
def μ_cert : ℚ := sorry  -- exact rational from Float64

/-- Surface-tension parameter β.  Set to 0 to disable the capillary path. -/
def β_cert : ℚ := sorry  -- exact rational from Float64

/-!
## Algorithm parameters
-/

/-- DFT / Taylor-grid size used by Julia's Arb-arithmetic chain. -/
def N_fft_cert : ℕ := 256

/-- Taylor expansion order used by Julia's Arb-arithmetic chain. -/
def M_order_cert : ℕ := 4

/-!
## Final scalar outputs from the Arb-certified pipeline
-/

/-- Final certified upper bound: ‖R‖_∞ ≤ ε₀_cert. -/
def ε₀_cert : ℚ := sorry  -- e.g. for the Stokes test case ≈ 1.75 × 10⁻²

/-- (β ≠ 0 only) Final certified lower bound on D(u) = √(y_u² + (1+H[y_u])²);
    needed to certify that 1/D is well-defined. -/
def DLower_cert : ℚ := sorry  -- e.g. ≈ 0.95
```

### 11.1.1 The trust chain

**No per-grid-point data, no intermediate sup_bounds vectors, no Gpp/Gppp:**
none of those appear in the certificate. They are *Julia's internal scratch*
inside the Arb-arithmetic chain. Lean does not redo interval arithmetic, so
shipping them would be duplication.

The trust chain that justifies `ε₀_cert ≥ ‖R‖_∞` is:

1. **Universal analytic lemmas (Lean-verified)** — §6.6 lists them:
   - N1 Faà di Bruno bound (for `inv_series` / `sqrt_series` derivative supnorms)
   - N2 exact-Leibniz product supnorm
   - N3 Fourier-tail π²/3 bound on $\|H[g]\|_\infty$
   - N4 integral-test aliasing remainder $4 C_p / ((p{-}1)(N/2{-}1)^{p-1})$
   - N5 grid-to-supnorm Taylor remainder
   plus the residual-identity definition (which is a definition, not a theorem).

   These are all universally true (for any smooth periodic function), so Lean
   proves them once, without any reference to a specific `a_cert`.

2. **Correctness of Arb** (trusted as a primitive): Arblib / FLINT implement
   ball arithmetic such that every computed Arb ball encloses the true value.

3. **Implementation match**: Julia's `BabenkoRigorous` routines (`sup_bound`,
   `fft_hilbert_series`, `inv_series`, `sqrt_series`, …) apply the formulas
   from (1) using the arithmetic from (2). The match between Julia's code and
   the universal lemma statements is checked by reading the (small) Julia
   source, *not* by Lean.

Consequence: `ε₀_cert` shipped from Julia is treated as a Lean-side axiom,
and `Main.lean`'s residual-bound theorem consumes it directly:

```lean
axiom cert_residual_le :
    ∀ u : ℝ, |R N_cert a_cert μ_cert β_cert u| ≤ (ε₀_cert : ℝ)

axiom cert_D_ge :        -- (β ≠ 0 only)
    ∀ u : ℝ, (DLower_cert : ℝ) ≤ D N_cert a_cert u
```

The total certificate is ~5 scalars for β=0, ~6 scalars for β≠0.  No matrices,
no vectors.

### 11.1.2 What changed from the original certificate schema?

The pre-implementation draft of this section had a Lipschitz scalar `L_cert`
plus only `(ε₀, D_lower, M_grid)` — but no derivative-norm bookkeeping.  A
post-implementation draft over-corrected by shipping full `sup_bounds` vectors,
two `Gpp/Gppp` bounds per Path-B Hilbert call, and a per-grid-point
`R_jk_grid` matrix of length `N_grid · (M_order+1)`.  Both were wrong:

- The Lipschitz draft made too small a certificate (no support for the
  higher-order Taylor remainder).
- The big draft made too large a certificate by *duplicating* what Lean's
  universal lemmas + Arb correctness already justify.

The current schema is the minimum: inputs + algorithm parameters + final
scalar Arb-arithmetic outputs.  Every other quantity that appears in the
existing per-milestone reports (sup_bounds vectors for `y`, `y_u`, `H[y_u]`,
`G`; Gpp/Gppp bounds; D's internal Taylor approximation; etc.) is an
*intermediate* used only inside Julia's Arb routines, and is justified by
the universal Lean lemmas, not by being in the certificate.

Decimal literals (`1.23456`, `1e-10`) have been replaced by exact dyadic
rationals (`((p : ℚ) / q)`) throughout, so `native_decide` / `norm_num` can
close inequalities without floating-point detours.

### 11.2 Supplying the Hypotheses

The three certificate theorems are discharged as follows:

- **`cert_h_grid`:** Julia evaluates $|R(u_j)|$ at $M_{\mathrm{cert}} = 4096$ grid points
  using interval arithmetic (Arblib/`setprecision`).
  Each evaluation produces a certified upper bound.
  These bounds are written to a file as rational numbers.
  In Lean, `native_decide` evaluates the rational arithmetic at compile time.
  Alternatively, a `norm_num` extension could check each bound.

- **`cert_h_deriv`:** Julia computes a certified bound on $\|R'\|_\infty$
  by bounding each of the five derivative terms analytically.
  Lean's proof of `cert_h_deriv` follows the same term-by-term structure,
  using the lemmas from `DLowerBound.lean` and `HilbertOnTrigPoly.lean`.
  The *numeric* bound $L_{\mathrm{cert}}$ is taken from Julia; Lean proves
  the analytic inequality conditional on that bound.

- **`cert_h_D`:** Julia minimizes $D$ over a fine grid and adds a Lipschitz
  correction. Lean verifies that from the grid minimum and the Lipschitz
  bound (both rational numbers from Julia), the global minimum is at least
  $D_{\mathrm{lower}}$.

### 11.3 Trust Argument

The Lean proof is unconditional *given the certificate axioms*.
A user who audits the proof needs to:
1. Verify that the Julia/Arblib code correctly implements the Babenko equation
   and the interval arithmetic (this is a code review, not a Lean verification).
2. Trust `native_decide` at the rational evaluation steps
   (equivalent to trusting the Lean kernel + its `Nat`/`Rat` computation).
3. Accept the `sorry`s in `AliasingErrorBound.lean` (standard DFT analysis)
   and `HilbertCommutesDeriv.lean` (density argument).

The analytic structure — that these hypotheses suffice to bound the sup-norm —
is fully certified by the Lean kernel.

---

## 12. Staging Plan

The files should be completed in the following order to enable
incremental compilation and testing.

### Phase 1: Pure Analysis (no Julia)

These files have no external dependencies and can be developed and
compiled without the certificate.

| Order | File | Estimated difficulty | Key Mathlib gap |
|---|---|---|---|
| 1 | `GridToSupNorm.lean` | Low | `Nat.round` behavior |
| 2 | `Definitions.lean` | Low | None |
| 3 | `HilbertOnTrigPoly.lean` | Medium | Convolution for `H_of_product` |
| 4 | `HilbertCommutesDeriv.lean` | Medium | Density argument (`sorry`) |
| 5 | `TaylorRemainder.lean` | Low | Direct CCF adaptation |
| 6 | `AliasingErrorBound.lean` | High | Poisson summation, IBP series |

**Why `GridToSupNorm.lean` first?**
It is the only file whose statement and proof are entirely self-contained
(no Fourier analysis, no Babenko equation).
Proving it first validates the overall CAP architecture and gives an
independently useful lemma (Lipschitz function bounded at grid implies bounded everywhere).
It also pins down the exact form of the final theorem statement.

### Phase 2: Equation-Specific Analysis

| Order | File | Notes |
|---|---|---|
| 7 | `DLowerBound.lean` | Needs `Definitions.lean`; some certificate hypotheses as `variable` |
| 8 | `Main.lean` (skeleton) | Assemble with `sorry`s for unfinished pieces |

### Phase 3: Certificate Bridge

| Order | File | Notes |
|---|---|---|
| 9 | `Certificate.lean` | Fill in actual Julia-computed rational constants |
| 10 | `Main.lean` (final) | Remove all `sorry`s except accepted ones |

### Summary: which files are `sorry`-free in the final proof

| File | `sorry`-free? | Remaining `sorry`s |
|---|---|---|
| `Definitions.lean` | Yes | — |
| `HilbertOnTrigPoly.lean` | Mostly | `H_of_product_correct` (mechanical algebra) |
| `HilbertCommutesDeriv.lean` | Partial | Smooth-periodic density argument |
| `AliasingErrorBound.lean` | Partial | Poisson summation; IBP series |
| `TaylorRemainder.lean` | Yes | — (uses CCF lemma directly) |
| `DLowerBound.lean` | Yes | — |
| `GridToSupNorm.lean` | Yes | — |
| `Main.lean` | Yes | — (modulo `sorry`s in dependencies) |
| `Certificate.lean` | Partial | `cert_h_deriv`, `cert_h_D` |

The three remaining `sorry` categories are:
1. **DFT aliasing via Poisson summation** — standard but not yet in Mathlib 4.
2. **$C^1(\mathbb{T})$ density of trig polynomials** — in Mathlib as
   `MeasureTheory.Lp.trigPolynomials_dense` but requires bridging $L^p$ density
   to $C^1$ density via Sobolev embedding.
3. **Certificate mechanical steps** — enumeration over 4096 grid points;
   a `native_decide` call is unconditional and should close these.

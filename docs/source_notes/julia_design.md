# BabenkoRigorous.jl — Design Document

> **Status: Pre-implementation draft (stale — see milestone reports for current architecture).**
>
> This document is the *original* design draft, written before the Julia
> package was implemented. Several sections are out of date with respect to
> the current code:
>
> - `TaylorSeries` no longer carries a single scalar `F_bound`; it carries a
>   `sup_bounds :: Vector{Arb}` of length `M_order+2` (one supnorm per derivative
>   order), and arithmetic uses exact Leibniz / Faà di Bruno (via ArbSeries
>   composition) on the vector.
> - `DMinBound.jl` and `GGrid` are not separate files / types; `dmin_bound` lives
>   in `TaylorSeries.jl` and returns `(D_lower, ts_D)`, and `compute_G_taylor`
>   (in `GDerivatives.jl`) returns a `TaylorSeries`.
> - `fft_hilbert` and `fft_hilbert_series` take `Gpp_bound` / `Gppp_bound`
>   kwargs and the aliasing constant is `4·Cp/((p−1)·(N/2−1)^{p−1})`, not the
>   `2·ζ(p)/N^{p−1}` form quoted here.
> - The output `sup_bounds` of `fft_hilbert_series` uses the Fourier-tail bound
>   `‖G^{(m+2)}‖∞ · π²/3` — not a Lebesgue-constant approach.
> - `H[y·y_u]` is routed through Path B (DFT) in the residual, not Path A.
>
> The authoritative description of the implemented modules is in the per-milestone
> reports `BabenkoRigorous/docs/m1_report.tex` through `m5_report.tex`.  This
> document is preserved for the high-level architecture / two-paths discussion
> (§§1–3) but should not be relied on for module-level signatures or formulas.

## 1. Overview

`BabenkoRigorous.jl` is a Julia package that produces a **computer-assisted proof certificate** bounding the residual of Babenko's equation for steady periodic water waves. Given a candidate approximate solution encoded as a truncated even cosine series, the package rigorously bounds `‖R‖_∞` where `R = LHS − RHS` is the residual of the Babenko formulation. The output is a Lean 4 source snippet (`Certificate.lean`) that encodes all inputs as exact rational literals and asserts that the certified bound holds — ready to be discharged by the Lean verifier.

### The equation

The approximate solution is a finite even cosine series on `[0, 2π]`:

```
y(u) = Σ_{k=1}^{N} a_k cos(ku)
```

The residual is `R = LHS − RHS` where:

```
LHS = μ H[y_u]
RHS = y + y H[y_u] + H(y y_u) − β ∂_u(y_u/D) + β H[∂_u((1+H[y_u])/D)]
D(u) = √(y_u(u)² + (1 + H[y_u](u))²)
```

`H` is the periodic Hilbert transform: `H[cos(ku)] = sin(ku)`, `H[sin(ku)] = −cos(ku)` for `k ≥ 1`. The parameters `μ` (speed) and `β` (Bond number / surface tension) are given rational constants; `β = 0` reduces to the pure gravity case.

### Two computational pathways for H

**Path A — exact trig polynomial arithmetic.** All terms in which `H` is applied to a combination of `y`, `y_u`, `y·y_u`, and their exact derivatives are trig polynomials. These are handled by exact Fourier coefficient manipulation: no DFT, no error term, `O(N²)` cost.

**Path B — DFT with rigorous aliasing bound.** The term `G = (1+H[y_u])/D` involves the irrational denominator `D = √P` and is not a trig polynomial. Its Hilbert transform and derivatives are computed via a DFT on a fine grid (using Arblib's `AcbPoly` multiplication), and a rigorous aliasing error is added to every ball.

### Key commutation identity

`H` and `∂_u` commute on periodic functions: `(H[G])^{(m)} = H[G^{(m)}]`. This lets the Taylor series of `H[G]` at grid points be built level-by-level: apply Path B to `G^{(m)}` for each `m = 0, …, M+1`, where `M` is the Taylor order used in the final sup-norm bound.

### Connection to Lean

`Certificate.lean` is emitted after a successful run. It encodes `a_k`, `μ`, `β`, `ε₀`, `D_lower`, and `M_grid` as Lean 4 rational literals plus a `is_certified : Bool := true` assertion. A companion Lean 4 formalization file imports the certificate and closes the verification gap between the Julia interval computation and the formal proof.

---

## 2. Package Structure

```
BabenkoRigorous.jl/
├── Project.toml               # Dependencies and metadata
├── src/
│   ├── BabenkoRigorous.jl     # Top-level module: re-exports public API
│   ├── FourierSeries.jl       # TrigPoly struct; exact H, deriv, *, eval, supnorm
│   ├── DMinBound.jl           # Rigorous lower bound on D(u) = √(y_u² + (1+H[y_u])²)
│   ├── GDerivatives.jl        # Analytic grid-point values of G^{(m)} via quotient rule
│   ├── FFTHilbert.jl          # Path B: DFT Hilbert transform with aliasing error bound
│   ├── TaylorSeries.jl        # TaylorSeries struct (adapted from cap-ccf); +, *, deriv, eval
│   ├── ResidualBound.jl       # Full residual assembly and ‖R‖_∞ bound
│   └── Certificate.jl         # Serialize certificate to Certificate.lean
└── test/
    ├── test_fourier.jl        # Algebraic identity tests for FourierSeries
    ├── test_dmin.jl           # DMinBound smoke tests
    ├── test_fft_hilbert.jl    # FFTHilbert convergence and aliasing bound tests
    ├── test_taylor.jl         # TaylorSeries arithmetic correctness tests
    ├── test_residual.jl       # End-to-end Stokes wave residual test
    └── runtests.jl            # Test harness
```

---

## 3. Dependencies (`Project.toml`)

```toml
[deps]
Arblib       = "fb37089c-8514-4489-9461-98f9c8763369"  # Arb ball arithmetic via FLINT
FFTW         = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"  # Float64 pre-check FFTs only
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"  # ζ(p) for aliasing bound
Primes       = "27ebfcd6-29c5-5fa9-bf4e-7612d18f0571"   # optional, for N_fft selection
```

**Note on FFTW:** Arblib's `AcbPoly` multiplication serves as the rigorous FFT. FFTW is used only in the Float64 pre-check phase (see Section 8) to catch gross errors cheaply before the expensive interval computation.

---

## 4. Source Files

---

### 4.1 `FourierSeries.jl`

#### Purpose

Represents a real-valued trig polynomial

```
f(u) = (a_0/2) + Σ_{k=1}^{N} [a_k cos(ku) + b_k sin(ku)]
```

and implements all the exact algebraic operations needed for Path A. No floating-point error is introduced: all coefficients are Arb balls, and each operation is exact in exact interval arithmetic.

#### Data Structure

```julia
struct TrigPoly
    N      :: Int              # degree: nonzero modes are k = 0..N
    a      :: Vector{Arb}      # a[k+1] = coefficient of cos(ku), k = 0..N
    b      :: Vector{Arb}      # b[k+1] = coefficient of sin(ku), k = 0..N
    prec   :: Int              # Arb working precision (bits)
end
```

**Invariants:**
- `length(a) == length(b) == N + 1`
- `a[1]` is the constant term (k=0); `b[1]` is always zero (sin(0)=0)
- All `Arb` balls have `precision == prec`

A pure cosine series (as the approximate solution `y`) satisfies `b[k+1] = 0` for all `k`, but general `TrigPoly` values arising from products will have both parts.

#### Key Functions

---

**`hilbert(f::TrigPoly) -> TrigPoly`**

Applies the periodic Hilbert transform mode-by-mode:

```
H[cos(ku)] = sin(ku)      for k ≥ 1
H[sin(ku)] = −cos(ku)     for k ≥ 1
H[1]       = 0
```

Algorithm (O(N)):

```julia
function hilbert(f::TrigPoly)
    N, p = f.N, f.prec
    a_out = Vector{Arb}(undef, N + 1)
    b_out = Vector{Arb}(undef, N + 1)
    a_out[1] = Arb(0; prec=p)   # H[const] = 0
    b_out[1] = Arb(0; prec=p)
    for k in 1:N
        # H maps: a_k cos(ku) + b_k sin(ku)  ->  a_k sin(ku) - b_k cos(ku)
        a_out[k+1] = -f.b[k+1]
        b_out[k+1] =  f.a[k+1]
    end
    return TrigPoly(N, a_out, b_out, p)
end
```

No error is incurred; this is exact on exact Arb coefficient vectors.

---

**`deriv(f::TrigPoly) -> TrigPoly`**

Differentiates: `∂_u[cos(ku)] = −k sin(ku)`, `∂_u[sin(ku)] = k cos(ku)`.

```julia
function deriv(f::TrigPoly)
    N, p = f.N, f.prec
    a_out = [Arb(0; prec=p) for _ in 0:N]
    b_out = [Arb(0; prec=p) for _ in 0:N]
    for k in 1:N
        a_out[k+1] = -Arb(k; prec=p) * f.b[k+1]   # d/du [b_k sin(ku)] = k b_k cos(ku) ? no
        b_out[k+1] =  Arb(k; prec=p) * f.a[k+1]   # d/du [a_k cos(ku)] = -k a_k sin(ku)
    end
    # Correct signs:
    # d/du [a_k cos(ku)] = -k a_k sin(ku)  -> contributes to b_out: -k a_k
    # d/du [b_k sin(ku)] =  k b_k cos(ku)  -> contributes to a_out: +k b_k
    return TrigPoly(N, a_out, b_out, p)
end
```

Explicitly:

```
(∂f)_a[k] = k · f.b[k]      (cos coefficient of f' comes from sin terms)
(∂f)_b[k] = −k · f.a[k]     (sin coefficient of f' comes from cos terms)
```

---

**`*(f::TrigPoly, g::TrigPoly) -> TrigPoly`**

Computes the pointwise product using the product-to-sum identities:

```
cos(ku) cos(lu) = (1/2)[cos((k−l)u) + cos((k+l)u)]
sin(ku) sin(lu) = (1/2)[cos((k−l)u) − cos((k+l)u)]
cos(ku) sin(lu) = (1/2)[sin((k+l)u) − sin((k−l)u)]   (for k≠l)
cos(ku) sin(ku) = (1/2) sin(2ku)
```

The output degree is `f.N + g.N`. Algorithm is an O(N²) Cauchy-type product over modes:

```julia
function Base.:*(f::TrigPoly, g::TrigPoly)
    M = f.N + g.N
    p = f.prec
    a_out = [Arb(0; prec=p) for _ in 0:M]
    b_out = [Arb(0; prec=p) for _ in 0:M]
    half = Arb(1; prec=p) / Arb(2; prec=p)

    for k in 0:f.N, l in 0:g.N
        ak, bk = f.a[k+1], f.b[k+1]
        al, bl = g.a[l+1], g.b[l+1]

        # cos(ku)*cos(lu) -> (1/2)[cos(|k-l|u) + cos((k+l)u)]
        cc = half * ak * al
        add_cos!(a_out, abs(k-l), cc, p)
        add_cos!(a_out, k+l,      cc, p)

        # sin(ku)*sin(lu) -> (1/2)[cos(|k-l|u) - cos((k+l)u)]
        ss = half * bk * bl
        add_cos!(a_out, abs(k-l),  ss, p)
        add_cos!(a_out, k+l,      -ss, p)

        # cos(ku)*sin(lu) - sin(ku)*cos(lu) -> sin((l-k)u) terms
        # cos(ku)*sin(lu) -> (1/2)[sin((k+l)u) + sin((l-k)u)]  (l-k can be negative)
        cs = half * ak * bl
        add_sin!(b_out, k+l,   cs, p)
        add_sin_signed!(b_out, l-k, cs, p)  # handles sign of negative index

        sc = half * bk * al
        add_sin!(b_out, k+l,   sc, p)
        add_sin_signed!(b_out, k-l, sc, p)
    end
    return TrigPoly(M, a_out, b_out, p)
end
```

where `add_sin_signed!(b, n, c, p)` adds `c` to `b[n+1]` when `n ≥ 0` and `−c` to `b[|n|+1]` when `n < 0` (using `sin(−nu) = −sin(nu)`), and the constant term `k=0` or `l=0` cases are handled by noting `sin(0)=0`, `cos(0)=1`.

**Correctness note:** The constant mode `k=0` in `f` contributes `a_0/2 · g(u)`, which is simply scaled addition. The factor of 1/2 in the convention `a_0/2` for the constant term must be tracked consistently: the standard convention stores `a[1] = a_0` and interprets the constant as `a[1]/2`, so the multiplication formula accounts for this.

---

**`eval(f::TrigPoly, u::Arb) -> Arb`**

Evaluates `f(u)` at a (possibly ball-valued) point via Clenshaw's algorithm or direct sum:

```julia
function eval(f::TrigPoly, u::Arb)
    p = f.prec
    result = f.a[1] / Arb(2; prec=p)   # constant term a_0/2
    for k in 1:f.N
        result += f.a[k+1] * cos(Arb(k; prec=p) * u) +
                  f.b[k+1] * sin(Arb(k; prec=p) * u)
    end
    return result
end
```

Arb's `cos` and `sin` are rigorous, so the output Arb ball rigorously encloses the true value. For ball inputs `u`, the ball radius of the trig functions is tracked automatically.

---

**`supnorm_bound(f::TrigPoly) -> Arb`**

Returns an upper bound on `‖f‖_∞` via the Bernstein L¹-of-coefficients estimate:

```
‖f‖_∞ ≤ |a_0|/2 + Σ_{k=1}^{N} (|a_k| + |b_k|)
```

This bound is trivial but rigorous and O(N):

```julia
function supnorm_bound(f::TrigPoly)
    p = f.prec
    s = abs(f.a[1]) / Arb(2; prec=p)
    for k in 1:f.N
        s += abs(f.a[k+1]) + abs(f.b[k+1])
    end
    return s
end
```

**When is this bound tight?** It equals `‖f‖_∞` exactly only when all terms have the same phase everywhere — for generic trig polynomials it can overestimate by a factor of `O(N)`. For the purpose of bounding `‖P'‖_∞` in `DMinBound.jl`, where `P'` is a sum of a moderate number of modes, this is adequate.

---

**Example usage:**

```julia
using Arblib, BabenkoRigorous

prec = 256
N = 8
# Build y = a_1 cos(u) + a_2 cos(2u) + a_3 cos(3u)
a = [Arb(0; prec=prec),
     Arb("0.12"; prec=prec),
     Arb("0.008"; prec=prec),
     Arb("0.0005"; prec=prec)]
b = [Arb(0; prec=prec) for _ in 0:3]
y = TrigPoly(3, a, b, prec)

yu   = deriv(y)        # y_u
Hyu  = hilbert(yu)     # H[y_u]
yyu  = y * yu          # y · y_u  (degree 6)
HyYu = hilbert(yyu)    # H[y · y_u]

println(supnorm_bound(Hyu))   # rigorous ‖H[y_u]‖_∞ bound
```

---

### 4.2 `DMinBound.jl`

#### Purpose

Produces a certified lower bound `D_min > 0` for

```
D(u) = √(y_u(u)² + (1 + H[y_u](u))²)
```

over `u ∈ [0, 2π]`. This lower bound is needed throughout `GDerivatives.jl` (to bound `1/D` and its powers) and in `ResidualBound.jl` (for the capillary terms). A positive `D_min` also certifies that the solution profile is non-self-intersecting.

#### Key Idea

Define `P(u) = y_u(u)² + (1 + H[y_u](u))²`. Then `P` is an exact `TrigPoly` (computed from `FourierSeries.jl`) and `D(u) = √P(u)`, so `D_min = √P_min`.

To lower-bound `P_min = min_{u ∈ [0,2π]} P(u)`:

1. Compute `P'(u) = deriv(P)` — exact `TrigPoly` of degree `2N`.
2. Bound `‖P'‖_∞ ≤ supnorm_bound(P')` — call this `L_P`.
3. Choose a grid of `M_grid` equally spaced points `u_j = 2πj/M_grid`, `j = 0, …, M_grid−1`.
4. Evaluate `P(u_j)` exactly (as Arb balls) for each `j`.
5. The minimum of `P` over any interval `[u_j, u_{j+1}]` of width `h = 2π/M_grid` satisfies:

```
min_{u ∈ [u_j, u_{j+1}]} P(u) ≥ P(u_j).lo − L_P · h
```

where `P(u_j).lo` is the lower endpoint of the Arb ball at the grid point. Hence:

```
P_min ≥ min_j [P(u_j).lo − L_P · (2π/M_grid)]
D_min = √(max(P_min, 0))
```

If `P_min ≤ 0` (which should not happen for a valid approximate solution), the run aborts with an error.

#### Data Structure

```julia
struct DMinResult
    P_min  :: Arb       # rigorous lower bound on min_u P(u)
    D_min  :: Arb       # rigorous lower bound on min_u D(u) = √P_min
    L_P    :: Arb       # Bernstein supnorm bound on P'
    M_grid :: Int       # number of grid points used
end
```

#### Key Function

```julia
function dmin_bound(y::TrigPoly; M_grid::Int=1024) -> DMinResult
    # Step 1: build P = y_u^2 + (1 + H[y_u])^2
    yu  = deriv(y)
    Hyu = hilbert(yu)
    one_plus_Hyu = add_const(Hyu, 1)   # TrigPoly: 1 + H[y_u]
    P   = yu * yu + one_plus_Hyu * one_plus_Hyu

    # Step 2: derivative bound
    dP  = deriv(P)
    L_P = supnorm_bound(dP)

    # Step 3: grid evaluation
    h   = Arb(2; prec=P.prec) * Arb(pi; prec=P.prec) / Arb(M_grid; prec=P.prec)
    P_min_lo = typemax(Float64)
    for j in 0:(M_grid - 1)
        u_j = h * Arb(j; prec=P.prec)
        Pj  = eval(P, u_j)
        lo  = Float64(Arblib.midref(Pj)) - Float64(Arblib.radref(Pj))
        lo -= Float64(Arblib.midref(L_P)) + Float64(Arblib.radref(L_P))
        lo -= Float64(Arblib.midref(h))   + Float64(Arblib.radref(h))
        P_min_lo = min(P_min_lo, lo)
    end

    P_min = Arb(P_min_lo; prec=P.prec)
    P_min ≤ 0 && error("DMinBound: P_min ≤ 0, approximate solution is invalid")
    D_min = sqrt(P_min)

    return DMinResult(P_min, D_min, L_P, M_grid)
end
```

**Choosing `M_grid`:** The tolerance `L_P · (2π/M_grid)` must be much smaller than the true gap `P_min − L_P·h`. In practice `M_grid = 4N` to `16N` suffices for moderate wave steepness; the function can be called twice with increasing `M_grid` until the bound is positive with sufficient margin.

---

### 4.3 `GDerivatives.jl`

#### Purpose

Computes, for each of the `M+2` derivative orders `m = 0, 1, …, M+1`, the value of

```
G^{(m)}(u_j) = ∂_u^m [(1 + H[y_u](u)) / D(u)]
```

at each grid point `u_j = 2πj/N_grid`, `j = 0, …, N_grid−1`. The result feeds directly into `FFTHilbert.jl` (Path B) and into the capillary term `cap1` in `ResidualBound.jl`.

`G` is not a trig polynomial (it involves `D = √P`), so its values at grid points are Arb balls, not exact Fourier coefficients.

#### Mathematical Structure

Write `G = Q/D` where:
- Numerator: `Q = 1 + H[y_u]` — an exact `TrigPoly`.
- Denominator: `D = √P` where `P = y_u² + (1+H[y_u])²` — not a trig polynomial.

Since `P = D²`, we have `P' = 2D D'`, so `D' = P'/(2D)`. More generally, all derivatives of `D` can be expressed in terms of `P` and its derivatives (which are exact trig polynomials) divided by powers of `D`.

Define the **rational form** at each grid point: maintain numerator polynomials (as Arb values) and the exact power of `D` in the denominator. At each step:

```
G^{(0)}(u_j) = Q(u_j) / D(u_j)

G^{(1)}(u_j) = [Q'(u_j) · D(u_j) − Q(u_j) · D'(u_j)] / D(u_j)²
             = [Q'(u_j) − G(u_j) · P'(u_j)/(2P(u_j)·D(u_j))] ... (simplified)
```

The cleanest recursive form uses the quotient rule and the identity `D' = P'/(2D)`:

```
G     = Q / D
G'    = Q'/D − Q·D'/D² = Q'/D − Q·P'/(2D³) = (Q' − G·P'/2) / D
G''   = [(Q' − G·P'/2)' − G'·P'/2] / D + [(Q' − G·P'/2)·(−D'/D²)]
```

In general, define `N_m(u_j)` to be the value such that `G^{(m)}(u_j) = N_m(u_j) / D^{2m−1}(u_j)` (for a suitable choice of denominator power that keeps the recurrence clean), or alternatively maintain a vector of Arb values `G^{(m)}(u_j)` directly via the quotient rule applied numerically at each grid point.

#### Data Structure

```julia
struct GGrid
    N_grid  :: Int
    M_order :: Int                        # derivatives 0..M_order stored
    u_pts   :: Vector{Arb}                # u_j = 2πj/N_grid, j=0..N_grid-1
    G_vals  :: Vector{Vector{Arb}}        # G_vals[m+1][j+1] = G^{(m)}(u_j)
    D_vals  :: Vector{Arb}                # D(u_j) at each grid point (for reuse)
    D_min   :: Arb                        # lower bound from DMinBound
    prec    :: Int
end
```

**Invariant:** `G_vals[m+1]` has length `N_grid` for `m = 0, …, M_order+1` (M+2 levels needed for aliasing error in `FFTHilbert`).

#### Algorithm

```julia
function compute_G_grid(y::TrigPoly, D_result::DMinResult,
                        M_order::Int; N_grid::Int=512) -> GGrid
    prec = y.prec
    yu   = deriv(y)
    Hyu  = hilbert(yu)
    Q    = add_const(Hyu, 1)     # Q = 1 + H[y_u]: TrigPoly
    P    = yu*yu + (add_const(Hyu,1)) * (add_const(Hyu,1))  # P: TrigPoly

    # Precompute TrigPoly derivatives of Q and P up to order M_order+1
    Q_derivs = [Q]
    P_derivs = [P]
    for m in 1:(M_order+1)
        push!(Q_derivs, deriv(Q_derivs[end]))
        push!(P_derivs, deriv(P_derivs[end]))
    end

    # Grid points
    h = Arb(2; prec=prec) * Arb(pi; prec=prec) / Arb(N_grid; prec=prec)
    u_pts = [h * Arb(j; prec=prec) for j in 0:(N_grid-1)]

    # Evaluate Q^{(m)} and P^{(m)} at each grid point (TrigPoly -> Arb)
    Qd_vals = [[eval(Q_derivs[m+1], u_pts[j+1]) for j in 0:(N_grid-1)]
               for m in 0:(M_order+1)]
    Pd_vals = [[eval(P_derivs[m+1], u_pts[j+1]) for j in 0:(N_grid-1)]
               for m in 0:(M_order+1)]

    # D(u_j) and 1/D(u_j) at grid points
    D_vals   = [sqrt(max(Pd_vals[1][j+1], D_result.P_min)) for j in 0:(N_grid-1)]
    invD     = [inv(D_vals[j+1]) for j in 0:(N_grid-1)]
    invD2    = [invD[j+1]^2 for j in 0:(N_grid-1)]

    # G^{(0)} = Q/D
    G_vals = Vector{Vector{Arb}}(undef, M_order + 2)
    G_vals[1] = [Qd_vals[1][j+1] * invD[j+1] for j in 0:(N_grid-1)]

    # G^{(m)} via iterated quotient rule: differentiate G^{(m-1)} = N_{m-1}/D
    # using the Leibniz rule and D' = P'/(2D):
    for m in 1:(M_order+1)
        Gm = Vector{Arb}(undef, N_grid)
        for j in 0:(N_grid-1)
            # (G^{(m-1)})' at u_j via the full quotient rule
            # G^{(m)}(u_j) = d/du G^{(m-1)}(u_j)
            # Computed by differentiating the analytic expression:
            # G^{(m)} = Σ_{r=0}^{m} C(m,r) Q^{(r)}· (1/D)^{(m-r)}
            # where (1/D)^{(k)} are the k-th derivatives of 1/D.
            # These are computed via the Faa di Bruno formula for (1/D)^{(k)},
            # which can be expressed in terms of P and its derivatives.
            # In practice, iterate the product rule on G = Q * (1/D).
            Gm[j+1] = apply_quotient_rule_step(G_vals[m], Qd_vals, Pd_vals,
                                                D_vals, invD, invD2, j, m, prec)
        end
        G_vals[m+1] = Gm
    end

    return GGrid(N_grid, M_order, u_pts, G_vals, D_vals, D_result.D_min, prec)
end
```

The helper `apply_quotient_rule_step` at order `m` implements:

```
G^{(m)}(u_j) = Σ_{r=0}^{m} C(m, r) · Q^{(r)}(u_j) · (1/D)^{(m-r)}(u_j)
```

where `(1/D)^{(k)}(u_j)` is the `k`-th derivative of `1/D` at `u_j`. Since `(1/D)' = −D'/(D²) = −P'/(2D³)`, higher derivatives follow from the chain and product rules applied to `P` (a known TrigPoly) and powers of `D`. In practice the simplest implementation is to maintain, at each grid point, a length-`(M+2)` vector `inv_D_derivs[k+1] = (1/D)^{(k)}(u_j)` computed level by level using:

```
(1/D)^{(k+1)} = − Σ_{r=0}^{k} C(k+1, r+1) · (1/D)^{(r)} · D^{(k-r+1)} / D
```

where `D^{(s)}(u_j)` is in turn computed from `P` and its derivatives.

**Error accounting:** Every Arb operation propagates ball radii automatically. The only non-rigorous step would be the use of `D_result.P_min` as a lower bound for `P(u_j)` inside `sqrt`; since `P_min` was certified by `DMinBound.jl`, `sqrt(P(u_j)) ≥ sqrt(P_min) = D_min > 0` is rigorous.

---

### 4.4 `FFTHilbert.jl`

#### Purpose

Implements Path B: given the values `{G^{(m)}(u_j)}` at `N_fft` equally spaced grid points `u_j = 2πj/N_fft`, compute `{H[G^{(m)}](u_j)}` with a rigorous total error bound. The function returns Arb balls whose radii include both the Arb rounding errors from the DFT arithmetic and the aliasing error from truncating the DFT to finitely many modes.

#### Algorithm

**Step 1 — Represent input as AcbVector.**

The `N_fft` grid values `G^{(m)}(u_j)` (already Arb balls) are wrapped as an `AcbPoly` with the values as coefficients. We use `N_fft` a power of 2 for FLINT's internal FFT dispatch.

**Step 2 — Compute DFT via AcbPoly multiplication.**

The DFT of a length-`N_fft` signal `f_j` can be computed as a polynomial evaluation at the `N_fft`-th roots of unity. The canonical way in Arblib is via `acb_poly_mul` (which FLINT dispatches to an FFT for large `N`). Concretely, the DFT is computed by multiplying the signal polynomial against a "selector" polynomial, following the same strategy as `fft_convolve!` in cap-ccf's `fft_convolution.jl`:

```julia
function fft_on_grid(vals::Vector{Arb}, N_fft::Int, prec::Int) -> Vector{Acb}
    buf = PolyMulBuffers(N_fft, prec)
    # Encode the DFT as a circular convolution:
    # Ĝ[k] = Σ_j G_j · exp(-2πijk/N_fft)
    # Achieved by polynomial multiplication + coefficient extraction.
    # ... (details: lift to AcbPoly, multiply, extract)
end
```

**Step 3 — Apply the Hilbert multiplier.**

In frequency domain, the discrete Hilbert transform multiplies the `k`-th DFT coefficient by `-i·sgn(k)` (with the convention `sgn(0) = 0`, `sgn(N_fft/2) = 0` for even `N_fft`):

```julia
for k in 0:(N_fft - 1)
    kk = k ≤ N_fft ÷ 2 ? k : k - N_fft   # centered frequency
    if kk == 0 || kk == N_fft ÷ 2
        F_hat[k+1] = Acb(0; prec=prec)
    else
        sign = kk > 0 ? -1 : 1
        # multiply by -i·sign = ±i (exact in interval arithmetic)
        F_hat[k+1] = Acb(0, Arb(sign; prec=prec); prec=prec) * G_hat[k+1]
    end
end
```

This multiplication is exact (multiplying by `±i` just swaps real and imaginary parts with a sign change).

**Step 4 — Compute IDFT.**

Apply the inverse DFT. Rounding errors from Steps 2–4 are tracked automatically by Arb ball arithmetic.

**Step 5 — Add aliasing error to each ball.**

#### Aliasing Error Bound

The `N_fft`-point DFT-based Hilbert transform `H_{N_fft}[G^{(m)}]` differs from the true `H[G^{(m)}]` due to aliasing. If `G^{(m)}` has Fourier coefficients `ĝ_k` decaying like `|ĝ_k| ≲ C/|k|^s`, then the aliasing tail satisfies:

```
|H[G^{(m)}](u_j) − H_{N_fft}[G^{(m)}](u_j)|
    ≤ 2 Σ_{l=1}^{∞} |ĝ_{lN_fft ± j}|
    ≤ 2 ‖G^{(m+p)}‖_∞ · ζ(p) · (2π)^{-p} / (N_fft/2)^{p-1} · (2π/N_fft)
```

More precisely, if `G^{(m)}` is `p`-times continuously differentiable with `‖G^{(m+p)}‖_∞ ≤ B_{m+p}`, then by the standard Fourier coefficient decay estimate:

```
|ĝ_k^{(m)}| ≤ B_{m+p} / (2π |k|^p)   for k ≠ 0
```

The aliasing error at grid point `u_j` is bounded by:

```
ε_alias(m) ≤ 2 · Σ_{r=1}^{∞} Σ_{s ∈ {-1,+1}} |ĝ^{(m)}_{rN_fft + s·j}|
           ≤ 2 · (B_{m+p} / (2π)) · Σ_{r=1}^{∞} Σ_{s} (rN_fft - N_grid/2)^{-p}
           ≤ B_{m+p} · 2 ζ(p) / (N_fft/2)^{p-1} · (2π)^{1-p} · (some geometric factor)
```

In the simplest practical form (suitable for an even, smooth `G^{(m)}`):

```
ε_alias(m) ≤ ‖G^{(m+p)}‖_∞ · 2 ζ(p) / N_fft^{p-1}
```

where `ζ(p) = Σ_{r=1}^{∞} r^{-p}` is the Riemann zeta function at `p`, and `p ≥ 2` is chosen smoothness order (e.g., `p = 6`).

**Where the formula comes from:** If the Fourier coefficients of `G^{(m)}` satisfy `|ĝ_k| ≤ C_p / |k|^p` with `C_p = ‖G^{(m+p)}‖_∞ / (2π)^p` (from integration by parts `p` times), then the aliasing contribution from all "wrap-around" modes at frequency `k` (i.e., from `k ± N_fft, k ± 2N_fft, …`) is bounded by `2C_p Σ_{r=1}^{∞} (rN_fft)^{-p} = 2C_p N_fft^{-p} ζ(p)`. Summing over the `N_fft/2` distinct positive frequencies gives the bound above.

**Connection to cap-ccf's Euler-Maclaurin formula:** In cap-ccf, the `_compute_discretization_error` function uses:

```
disc_error ≤ (|B_{2P}|/(2P)!) · dx^{2P} · 2L · sup|f^{(2P)}|
```

This is the Euler-Maclaurin remainder for the composite trapezoidal rule on `[-L, L]`. The periodic aliasing bound here is the analogous formula for the periodic setting: both express the error from discretizing a smooth convolution integral via a sum, both scale as `N^{-(p-1)}` or `dx^{2P}`, and both require a bound on a high-order derivative of the integrand. The main difference is that in the periodic setting the kernel (`H` on the circle) has no tail truncation issue, so only the aliasing term appears (no tail error). The formula `2ζ(p)/N_fft^{p-1}` plays the role of `(|B_{2P}|/(2P)!) · dx^{2P} · 2L`.

#### Implementation

```julia
function fft_hilbert(G_m_vals::Vector{Arb},        # G^{(m)}(u_j), j=0..N_fft-1
                     B_m_plus_p::Arb,               # bound on ‖G^{(m+p)}‖_∞
                     p::Int,                         # smoothness order (e.g. 6)
                     prec::Int) -> Vector{Arb}       # H[G^{(m)}](u_j) + aliasing ball

    N_fft = length(G_m_vals)
    @assert ispow2(N_fft) "N_fft must be a power of 2"

    # Aliasing error
    zeta_p   = Arb(zeta(p); prec=prec)              # ζ(p) via SpecialFunctions
    eps_alias = B_m_plus_p * Arb(2; prec=prec) * zeta_p /
                Arb(N_fft; prec=prec)^Arb(p-1; prec=prec)

    # Step 1-4: DFT, multiply by -i·sgn(k), IDFT via AcbPoly
    H_vals_acb = dft_hilbert_via_acbpoly(G_m_vals, N_fft, prec)

    # Step 5: extract real parts + add aliasing ball
    result = Vector{Arb}(undef, N_fft)
    for j in 1:N_fft
        re = real(H_vals_acb[j])
        Arblib.add_error!(re, eps_alias)
        result[j] = re
    end
    return result
end
```

The `sup_bound` on `‖G^{(m+p)}‖_∞` is provided by the caller from `GGrid.G_vals[m+p+1]` via a Bernstein-style grid maximum plus a Taylor remainder step (see `ResidualBound.jl`).

---

### 4.5 `TaylorSeries.jl`

#### Purpose

Represents a smooth function `f : [0, 2π] → ℝ` as a piecewise Taylor series at `N_grid` equally spaced grid points, with rigorous sup-norm bounds on all derivatives up to order `M+1`. This is adapted directly from cap-ccf's `taylor_series.jl` with two key differences:

1. Grid points are `u_j = 2πj/N_grid` on `[0, 2π]` (not `z_j` on `[-1,1]`).
2. No coordinate transformation (no `σ_C` rescaling, no two-sigma trick).

#### Data Structure

```julia
struct TaylorSeries
    u_points     :: Vector{Arb}          # u_j = 2πj/N_grid, j=0..N_grid-1
    taylor_coeffs :: Vector{Vector{Arb}} # [j+1][m+1] = f^{(m)}(u_j), raw derivatives
    taylor_order :: Int                   # M: coefficients stored for m = 0..M
    sup_bounds   :: Vector{Arb}          # [m+1] = rigorous bound on ‖f^{(m)}‖_∞, m=0..M+1
    prec         :: Int

    # Pre-computed caches
    factorial_cache  :: Dict{Int, Arb}
    binomial_cache   :: Dict{Tuple{Int,Int}, Arb}
end
```

**Convention (matching cap-ccf):** `taylor_coeffs[j+1][m+1] = f^{(m)}(u_j)` stores raw derivatives, NOT divided by `m!`. Division by `m!` happens at evaluation time inside `evaluate_on_subinterval`.

**Invariant:** `length(taylor_coeffs) == N_grid`, each inner vector has length `M+1` (indices `1..M+1` for orders `0..M`), and `length(sup_bounds) == M+2` (includes the `(M+1)`-th derivative bound needed for Taylor remainders).

#### Key Operations

---

**`evaluate_on_subinterval(ts::TaylorSeries, u::Arb) -> Arb`**

Evaluates `f(u)` rigorously for a (possibly ball-valued) `u ∈ [0, 2π]`. Find the nearest-left grid point `u_j`, compute the Taylor sum, and add a rigorous remainder:

```julia
function evaluate_on_subinterval(ts::TaylorSeries, u::Arb)
    p = ts.prec

    # Find nearest-left grid point index
    u_mid = Float64(Arblib.midref(u))
    N = length(ts.u_points)
    idx = max(1, searchsortedfirst(Float64[Float64(Arblib.midref(p)) for p in ts.u_points],
                                    u_mid) - 1)

    u_j = ts.u_points[idx]
    h   = u - u_j

    # Taylor sum: Σ_{m=0}^{M} f^{(m)}(u_j) · h^m / m!
    taylor_sum = Arb(0; prec=p)
    h_power    = Arb(1; prec=p)
    for m in 0:ts.taylor_order
        taylor_sum += (ts.taylor_coeffs[idx][m+1] / factorial_arb(ts, m)) * h_power
        h_power    *= h
    end

    # Remainder: |R| ≤ ‖f^{(M+1)}‖_∞ · |h|^{M+1} / (M+1)!
    M1     = ts.taylor_order + 1
    h_ball = abs(u - u_j)
    if idx < N
        h_ball = Arblib.union(h_ball, abs(u - ts.u_points[idx+1]))
    end
    remainder = ts.sup_bounds[M1+1] * h_ball^M1 / factorial_arb(ts, M1)
    return add_error(taylor_sum, abs(remainder))
end
```

**Periodicity:** for `u` near `2π`, wrapping is handled by treating the grid as covering `[0, 2π)` with the last interval `[u_{N-1}, 2π]` implicitly covered.

---

**`derivative(ts::TaylorSeries) -> TaylorSeries`**

Shifts the coefficient array: `(f')^{(m)}(u_j) = f^{(m+1)}(u_j)`. Shifts `sup_bounds` similarly:

```julia
function derivative(ts::TaylorSeries)
    N = length(ts.u_points)
    new_coeffs = [[ts.taylor_coeffs[j][m+2] for m in 0:(ts.taylor_order-1)]
                  for j in 1:N]
    new_bounds = ts.sup_bounds[2:end]
    return TaylorSeries(ts.u_points, new_coeffs, new_bounds)
end
```

After differentiation, `taylor_order` drops by 1 and `length(sup_bounds)` drops by 1.

---

**`*(ts1::TaylorSeries, ts2::TaylorSeries) -> TaylorSeries`**

Leibniz product rule: `(fg)^{(k)} = Σ_{j=0}^{k} C(k,j) f^{(j)} g^{(k-j)}`.

```julia
function Base.:*(a::TaylorSeries, b::TaylorSeries)
    M = min(a.taylor_order, b.taylor_order)
    N = length(a.u_points)
    p = a.prec

    new_coeffs = Vector{Vector{Arb}}(undef, N)
    for i in 1:N
        c = Vector{Arb}(undef, M+1)
        for k in 0:M
            s = Arb(0; prec=p)
            for j in 0:k
                s += binomial_arb(a, k, j) *
                     a.taylor_coeffs[i][j+1] * b.taylor_coeffs[i][k-j+1]
            end
            c[k+1] = s
        end
        new_coeffs[i] = c
    end

    # Sup bounds via Leibniz on bounds
    new_bounds = Vector{Arb}(undef, M+2)
    for k in 0:(M+1)
        s = Arb(0; prec=p)
        for j in 0:k
            j+1 ≤ length(a.sup_bounds) && k-j+1 ≤ length(b.sup_bounds) &&
                (s += binomial_arb(a, k, j) * a.sup_bounds[j+1] * b.sup_bounds[k-j+1])
        end
        new_bounds[k+1] = s
    end
    return TaylorSeries(a.u_points, new_coeffs, new_bounds)
end
```

---

**`+(ts1::TaylorSeries, ts2::TaylorSeries) -> TaylorSeries`**

Componentwise addition; bounds add by triangle inequality. Matches cap-ccf exactly.

---

**Constructing a `TaylorSeries` from a `TrigPoly`:**

```julia
function taylor_from_trigpoly(f::TrigPoly, M::Int;
                               N_grid::Int=512) -> TaylorSeries
    prec = f.prec
    h = Arb(2; prec=prec) * Arb(pi; prec=prec) / Arb(N_grid; prec=prec)
    u_pts = [h * Arb(j; prec=prec) for j in 0:(N_grid-1)]

    # Compute the first M+1 TrigPoly derivatives
    derivs = [f]
    for _ in 1:M
        push!(derivs, deriv(derivs[end]))
    end
    # One more for the remainder bound
    push!(derivs, deriv(derivs[end]))

    # Taylor coeffs: taylor_coeffs[j+1][m+1] = derivs[m+1](u_pts[j+1])
    coeffs = [[eval(derivs[m+1], u_pts[j+1]) for m in 0:M] for j in 0:(N_grid-1)]

    # Sup bounds via Bernstein (exact for trig polys; tight enough)
    bounds = [supnorm_bound(derivs[m+1]) for m in 0:(M+1)]

    return TaylorSeries(u_pts, coeffs, bounds)
end
```

---

### 4.6 `ResidualBound.jl`

#### Purpose

Assembles the full residual `R = LHS − RHS` as a `TaylorSeries` and returns a certified upper bound on `‖R‖_∞`.

#### Step-by-Step Assembly

---

**Step 1 — Build `TrigPoly y` from coefficients.**

```julia
function build_y(a_coeffs::Vector{Arb}, prec::Int) -> TrigPoly
    N = length(a_coeffs)
    a_full = [Arb(0; prec=prec); a_coeffs]   # a_full[k+1] = a_k, k=0..N
    b_full = [Arb(0; prec=prec) for _ in 0:N]
    return TrigPoly(N, a_full, b_full, prec)
end
```

---

**Step 2 — Path A terms (all exact TrigPolys).**

```julia
y       = build_y(a_coeffs, prec)
yu      = deriv(y)                        # y_u
Hyu     = hilbert(yu)                     # H[y_u]
y_Hyu   = y * add_const(Hyu, 0)          # y · H[y_u]  (TrigPoly, degree 2N)
yyu     = y * yu                          # y · y_u     (TrigPoly, degree 2N)
H_yyu   = hilbert(yyu)                    # H[y·y_u]

# μ H[y_u]: the LHS
mu_Hyu  = scalar_mul(Hyu, mu)
```

Convert each Path A TrigPoly to a `TaylorSeries` at the grid points:

```julia
M = 6   # Taylor order
ts_mu_Hyu  = taylor_from_trigpoly(mu_Hyu,  M; N_grid=N_grid)
ts_y       = taylor_from_trigpoly(y,        M; N_grid=N_grid)
ts_y_Hyu   = taylor_from_trigpoly(y_Hyu,   M; N_grid=N_grid)
ts_H_yyu   = taylor_from_trigpoly(H_yyu,   M; N_grid=N_grid)
```

For each derivative order `m`, `ts_y.taylor_coeffs[j+1][m+1] = y^{(m)}(u_j)` — the exact `m`-th derivative of the trig polynomial `y` evaluated at grid point `u_j`. No approximation error enters here; the Arb ball radius is purely from the working-precision arithmetic on the coefficients.

---

**Step 3 — Compute `D_min` via `DMinBound.jl`.**

```julia
D_result = dmin_bound(y; M_grid=4*N_grid)
D_min    = D_result.D_min
```

---

**Step 4 — Path B terms (β ≠ 0).**

Skip entirely if `β = 0` (gravity wave, no surface tension).

```julia
if !iszero(beta)
    # Compute G^{(m)} at N_grid points for m = 0..M+1
    gg = compute_G_grid(y, D_result, M + 1; N_grid=N_grid)

    # For each m, compute H[G^{(m)}] via FFTHilbert
    p_smooth = 6   # smoothness order for aliasing bound
    HG_vals = Vector{Vector{Arb}}(undef, M + 2)
    for m in 0:(M+1)
        # Bound on ‖G^{(m+p)}‖_∞ from the grid data + Taylor remainder
        B_m_plus_p = grid_supnorm_bound(gg.G_vals[m + p_smooth + 1],
                                         gg.sup_bounds[m + p_smooth + 1])
        HG_vals[m+1] = fft_hilbert(gg.G_vals[m+1], B_m_plus_p, p_smooth, prec)
    end

    # cap1^{(m)}(u_j) = ∂^m[∂_u(y_u/D)](u_j)  -- from GDerivatives (analytic)
    cap1_vals = compute_cap1_grid(gg, M)   # uses quotient rule on y_u/D

    # cap2^{(m)}(u_j) = H[G^{(m+1)}](u_j)    -- from HG_vals shifted by 1
    cap2_vals = [HG_vals[m+2] for m in 0:M]
end
```

---

**Step 5 — Assemble the residual `TaylorSeries`.**

At each grid point `u_j` and each derivative order `m`:

```
R^{(m)}(u_j) = μ·(H[y_u])^{(m)}(u_j)
             − y^{(m)}(u_j)
             − (y · H[y_u])^{(m)}(u_j)      [Leibniz: from ts_y * ts_Hyu product]
             − (H[y · y_u])^{(m)}(u_j)
             + β · cap1^{(m)}(u_j)
             − β · cap2^{(m)}(u_j)
```

The `(y · H[y_u])^{(m)}` term is most naturally computed via `TaylorSeries` multiplication (using the Leibniz rule, as in cap-ccf). Define:

```julia
ts_Hyu_exact = taylor_from_trigpoly(Hyu, M; N_grid=N_grid)
ts_prod_yHyu = ts_y * ts_Hyu_exact   # Leibniz product
```

Then assemble:

```julia
R_ts = ts_mu_Hyu - ts_y - ts_prod_yHyu - ts_H_yyu
if !iszero(beta)
    ts_cap1 = TaylorSeries_from_grid(cap1_vals, u_pts, sup_cap1)
    ts_cap2 = TaylorSeries_from_grid(cap2_vals, u_pts, sup_cap2)
    R_ts = R_ts + beta * ts_cap1 - beta * ts_cap2
end
```

---

**Step 6 — Compute `‖R‖_∞`.**

Use the Taylor remainder formula: for `u ∈ [u_j, u_{j+1}]` with `h = 2π/N_grid`,

```
|R(u)| ≤ |R(u_j)| + ‖R^{(M+1)}‖_∞ · h^{M+1} / (M+1)!
```

Taking the max over all grid points:

```
‖R‖_∞ ≤ max_j |R(u_j)|.hi + ‖R^{(M+1)}‖_∞ · (2π/N_grid)^{M+1} / (M+1)!
```

where `.hi` denotes the upper endpoint of the Arb ball:

```julia
function residual_sup_bound(R_ts::TaylorSeries) -> Arb
    prec  = R_ts.prec
    M1    = R_ts.taylor_order + 1
    h     = Arb(2; prec=prec) * Arb(pi; prec=prec) / Arb(length(R_ts.u_points); prec=prec)

    # Grid maximum
    max_at_grid = Arb(0; prec=prec)
    for j in 1:length(R_ts.u_points)
        val = R_ts.taylor_coeffs[j][1]   # R(u_j)
        upper = abs(Arb(Arblib.midref(val); prec=prec)) + Arb(Arblib.radref(val); prec=prec)
        upper > max_at_grid && (max_at_grid = upper)
    end

    # Taylor remainder contribution
    sup_deriv = R_ts.sup_bounds[M1 + 1]   # ‖R^{(M+1)}‖_∞
    remainder = sup_deriv * h^M1 / Arb(factorial(big(M1)); prec=prec)

    return max_at_grid + remainder
end
```

---

### 4.7 `Certificate.jl`

#### Purpose

Serializes the certificate to a Lean 4 source file (`Certificate.lean`) containing all inputs as exact rational literals and the certified bound as a verified constant.

#### What is Written

The certificate file encodes:
- The input coefficients `a_k` as exact rational literals (the Arb midpoints, converted to nearby rationals within the certified error margin).
- `μ`, `β` as rational literals.
- `ε₀`: the certified upper bound on `‖R‖_∞` (Arb upper endpoint, expressed as a rational).
- `D_lower`: the certified lower bound on `min_u D(u)`.
- `M_grid`, `N_grid`, `M_order` used in the computation.
- `is_certified : Bool := true`.

#### Example Generated `Certificate.lean`

```lean
-- Auto-generated by BabenkoRigorous.jl
-- Do not edit by hand.

namespace BabenkoWaveCertificate

/-- Degree of the approximate solution -/
def N : ℕ := 8

/-- Approximate solution coefficients a_k for k = 1..N -/
def a_coeffs : List ℚ :=
  [ 1219342598729343/10000000000000000,   -- a_1  ≈ 0.12193...
    7891234561234567/1000000000000000000, -- a_2  ≈ 0.00789...
    4512345678901234/10000000000000000000,-- a_3  ≈ 0.000451...
    1234567890123456/100000000000000000000,-- a_4
    3456789012345678/1000000000000000000000,-- a_5
    9012345678901234/10000000000000000000000,-- a_6
    2345678901234567/100000000000000000000000,-- a_7
    6789012345678901/1000000000000000000000000 ]-- a_8

/-- Wave speed parameter μ -/
def mu : ℚ := 12193425987293/10000000000000

/-- Bond number (surface tension parameter) β = 0 for gravity waves -/
def beta : ℚ := 0

/-- Certified upper bound on ‖R‖_∞ -/
def epsilon_0 : ℚ := 1/10000000000   -- ε₀ < 10⁻¹⁰

/-- Certified lower bound on min_u D(u) -/
def D_lower : ℚ := 8743219876543/10000000000000   -- D_lower ≈ 0.874...

/-- Grid parameters used in certification -/
def M_grid  : ℕ := 4096
def N_grid  : ℕ := 512
def M_order : ℕ := 6

/-- Certificate validity flag -/
def is_certified : Bool := true

end BabenkoWaveCertificate
```

The companion Lean file imports this and states a theorem of the form:

```lean
theorem babenko_residual_bound :
    ε₀ = BabenkoWaveCertificate.epsilon_0 ∧
    ‖R(a_coeffs, mu, beta)‖_∞ ≤ ε₀ := by
  decide  -- closes if is_certified = true
```

(The full Lean formalization of the functional-analytic argument that small `‖R‖_∞` implies existence of a nearby exact solution is a separate effort; this certificate provides the numerical input.)

#### Julia API

```julia
function write_certificate(outpath::String,
                            a_coeffs :: Vector{Arb},
                            mu       :: Arb,
                            beta     :: Arb,
                            eps0     :: Arb,
                            D_lower  :: Arb,
                            params   :: NamedTuple)
    # Convert Arb upper endpoints to exact rationals
    # Write the Lean 4 file
end
```

---

## 5. Error Budget

The following table summarizes every error source, where it enters the computation, and how it is bounded.

| # | Error Source | Location | Bound |
|---|---|---|---|
| 1 | Arb ball arithmetic (rounding in all Arb operations) | Throughout | Propagated automatically by FLINT's interval arithmetic; no action needed |
| 2 | TrigPoly coefficient storage | `FourierSeries.jl` | Coefficients are exact Arb balls; no approximation |
| 3 | TrigPoly evaluation at grid points (`eval`) | `FourierSeries.jl` | Rigorous Arb `cos`/`sin` enclosures |
| 4 | `supnorm_bound` overestimate (Bernstein bound) | `FourierSeries.jl`, `DMinBound.jl` | Conservative but rigorous: `‖f‖_∞ ≤ Σ(|â_k| + |b̂_k|)` |
| 5 | `P_min` underestimate from grid discretization | `DMinBound.jl` | `P_min ≥ min_j P(u_j).lo − ‖P'‖_∞ · (2π/M_grid)` |
| 6 | `1/D` bounding with `D ≥ D_min` | `GDerivatives.jl` | `|1/D(u_j)| ≤ 1/D_min`, rigorous since `D_min` certified |
| 7 | Quotient-rule expansion for `G^{(m)}(u_j)` | `GDerivatives.jl` | All operations on Arb balls; no approximation beyond ball propagation |
| 8 | DFT rounding error (Steps 2–4 of FFT Hilbert) | `FFTHilbert.jl` | Tracked automatically by Arb `AcbPoly` multiplication |
| 9 | Aliasing error from finite DFT | `FFTHilbert.jl` | `ε_alias(m) ≤ ‖G^{(m+p)}‖_∞ · 2ζ(p) / N_fft^{p-1}` added to each ball |
| 10 | Taylor remainder in `evaluate_on_subinterval` | `TaylorSeries.jl` | `‖f^{(M+1)}‖_∞ · h^{M+1} / (M+1)!` added as ball error |
| 11 | Taylor remainder in final `‖R‖_∞` bound | `ResidualBound.jl` | Same formula applied to `R_ts` at the last step |
| 12 | Rational approximation of Arb midpoints in certificate | `Certificate.jl` | Rational chosen within Arb ball radius; certificate encodes the outer rational bound |

**Dominant error sources in practice:** (9) aliasing error and (5) `P_min` discretization error. Both decay rapidly with `N_fft` and `M_grid` respectively. For moderate wave steepness (`a_1 ≈ 0.1`), `N_fft = 512` and `p = 6` give aliasing errors below `10^{-15}` at double-precision level, comfortably below the `10^{-10}` certification target.

---

## 6. Testing Strategy

### Float64 Pre-Check

Before the expensive Arb computation, run a Float64 residual evaluation:

```julia
function float64_precheck(a_coeffs::Vector{Float64}, mu::Float64, beta::Float64;
                           N_grid::Int=1024) -> Float64
    # Fast FFTW-based evaluation of R at N_grid points
    # Returns max|R| in Float64
    # If max|R| > 1e-3, warn user: approximate solution may be poor
end
```

This catches gross errors (wrong sign of `mu`, missing factor of 2π in H, etc.) in milliseconds and avoids wasting hours on a bad approximate solution.

### Low-Precision Arb Smoke Test

Run the full Arb pipeline at `prec = 64` (rather than the production `prec = 256`) with a known small example (N=4 Stokes wave). Verify:
- `is_certified = true`
- `ε₀ < 1e-4`
- Total runtime < 30 seconds

### Algebraic Identity Tests (`test_fourier.jl`)

These test the exact operations in `FourierSeries.jl`:

1. **Hilbert involution:** `H(H(f)) = −f` for all trig polys (up to constant term).
2. **H and deriv commute:** `H(∂f) = ∂(H(f))`.
3. **Product degree:** `(f * g).N == f.N + g.N`.
4. **Product-to-sum identity check:** for `f = cos(2u)`, `g = cos(3u)`, verify `f*g` has the expected Fourier coefficients for `cos(u)` and `cos(5u)` with coefficient `1/2`.
5. **Parseval check:** `‖f‖_L² = (π Σ_k (a_k² + b_k²))^{1/2}` via coefficient formula, compared to numerical integration.
6. **`supnorm_bound` rigorousness:** for several random trig polys, verify `supnorm_bound(f) ≥ max(|f(u_j)|)` at a fine grid.

### `DMinBound.jl` Tests

1. For `y = 0` (flat surface), `D(u) ≡ 1`, so `D_min = 1` exactly. Verify.
2. For a small-amplitude `y = ε cos(u)`, verify `D_min ≥ 1 − 2ε` and the certified bound is within `O(ε²)` of the true minimum.
3. Verify that `M_grid` refinement monotonically improves (increases) the lower bound.

### `FFTHilbert.jl` Tests

1. **Known trig polynomial:** apply `fft_hilbert` to `f(u) = cos(3u)`. The exact answer is `sin(3u)`. Verify that the returned Arb balls contain the exact values at grid points.
2. **Aliasing bound decreases with `N_fft`:** run `fft_hilbert` at `N_fft = 64, 128, 256, 512` and verify `ε_alias(m)` decreases as `N_fft^{-(p-1)}`.
3. **Consistency with TrigPoly hilbert:** for `G = H[y_u]/D_const` (a trig poly divided by a constant), verify `fft_hilbert` and `hilbert` agree within the aliasing bound.

### End-to-End Stokes Wave Test (`test_residual.jl`)

Use the well-known Stokes wave coefficients at steepness `ka ≈ 0.3` (available in the literature to many decimal places). At `N = 16`, `mu` chosen to satisfy the dispersion relation:

1. Run the full pipeline.
2. Verify `is_certified = true`.
3. Verify `ε₀ < 10^{-8}` (achievable at this steepness with moderate precision).
4. Verify the output `Certificate.lean` is valid Lean 4 syntax (parse check).

---

## 7. Staging (Recommended Implementation Order)

| Stage | Files | What to implement | Milestone |
|---|---|---|---|
| 0 | `FourierSeries.jl` | `TrigPoly` struct, `hilbert`, `deriv`, `*`, `eval`, `supnorm_bound` | All algebraic identity tests pass |
| 1 | `DMinBound.jl` | `dmin_bound` with grid evaluation | `D_min` test for flat surface and small amplitude |
| 2 | `TaylorSeries.jl` | Port from cap-ccf: struct, `+`, `-`, `*`, `derivative`, `evaluate_on_subinterval`; adapt grid to `[0, 2π]` | `test_taylor.jl` passes |
| 3 | `FFTHilbert.jl` | DFT via `AcbPoly` multiplication (reuse `PolyMulBuffers` from cap-ccf), Hilbert multiplier, aliasing bound | Known trig poly test passes; aliasing scaling test passes |
| 4 | `GDerivatives.jl` | `GGrid` struct, `compute_G_grid` with quotient rule, `compute_cap1_grid` | Consistency test with `TrigPoly` hilbert for polynomial `G` |
| 5 | `ResidualBound.jl` | Full assembly Steps 1–6 with `β = 0` (gravity wave) | Float64 pre-check + Arb smoke test at low precision |
| 6 | `ResidualBound.jl` | Add capillary terms (`β ≠ 0`, Steps 4–5 complete) | Capillary wave test |
| 7 | `Certificate.jl` | Write `Certificate.lean`; rational conversion | Certificate syntax check; Lean import test |
| 8 | `BabenkoRigorous.jl` | Top-level module, public API, `Project.toml` | End-to-end Stokes wave test |
| 9 | Optimization | `PolyMulBuffers` reuse, Bernstein vs. grid supnorm, parallelism | Production-scale certification (`N = 64`) within acceptable time |

**Reuse note (Stage 3):** `PolyMulBuffers` and `fft_convolve!` from cap-ccf's `fft_convolution.jl` can be copied verbatim or registered as a shared internal utility. The key lesson from cap-ccf (per the file header) is to pre-allocate the three `AcbPoly` buffers once per thread rather than constructing fresh ones per call, to avoid FLINT thread-local pool growth. This applies directly to `FFTHilbert.jl`.

**Reuse note (Stage 2):** The `TaylorSeries` struct, `+`, `-`, `*`, `derivative`, and `evaluate_on_rbf` from cap-ccf's `taylor_series.jl` are directly portable. The main adaptation is replacing `z_points` on `[-1,1]` with `u_points` on `[0, 2π]`, and removing the `σ_C` coordinate transformation. The `shift_taylor` and `recoordinate` (Faà di Bruno) functions are not needed in the periodic setting.

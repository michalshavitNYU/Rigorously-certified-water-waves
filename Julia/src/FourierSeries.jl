"""
FourierSeries.jl — Exact trig-polynomial arithmetic using Arblib ball arithmetic.

Convention:
  f(u) = a[1]/2 + Σ_{k=1}^{N} [a[k+1]*cos(ku) + b[k+1]*sin(ku)]

where a[1] is the DC (constant) coefficient (stored as the full value; the 1/2
factor is applied at evaluation time), and b[1] = 0 always.
"""

using Arblib

# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------

struct TrigPoly
    a    :: Vector{Arb}   # a[k+1] = coefficient of cos(ku), k=0..N; a[1] = DC term
    b    :: Vector{Arb}   # b[k+1] = coefficient of sin(ku), k=0..N; b[1] = 0 always
    prec :: Int
end

"""
    degree(f::TrigPoly) -> Int

Return the degree N of the trig polynomial (highest mode index).
"""
degree(f::TrigPoly) = length(f.a) - 1

# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

"""
    TrigPoly(a::Vector{Float64}, b::Vector{Float64}; prec=128)

Construct a TrigPoly from Float64 coefficient vectors, converting each to Arb.
`a[1]` = DC coefficient (interpreted as a[1]/2 in the series), `a[k+1]` = cos(ku) coefficient.
`b[1]` should be 0; `b[k+1]` = sin(ku) coefficient.
"""
function TrigPoly(a::Vector{Float64}, b::Vector{Float64}; prec::Int=128)
    @assert length(a) == length(b) "a and b must have the same length"
    a_arb = [Arb(x; prec=prec) for x in a]
    b_arb = [Arb(x; prec=prec) for x in b]
    return TrigPoly(a_arb, b_arb, prec)
end

"""
    TrigPoly(a::Vector{<:Real}, b::Vector{<:Real}, prec::Int)

Construct from generic real vectors (including Arb).
"""
function TrigPoly(a::Vector{<:Real}, b::Vector{<:Real}, prec::Int)
    a_arb = [Arb(x; prec=prec) for x in a]
    b_arb = [Arb(x; prec=prec) for x in b]
    return TrigPoly(a_arb, b_arb, prec)
end

"""
    cosine_series(coeffs::Vector{<:Real}; prec=128) -> TrigPoly

Build a pure cosine series where `coeffs[k]` is the coefficient of `cos((k-1)*u)`.
That is, `coeffs[1]` is the DC coefficient, `coeffs[2]` is cos(u), etc.

Alternatively, if you want coeffs[k] = coefficient of cos(k*u) for k=0,1,...,
pass a 1-indexed vector where coeffs[1] is for cos(0*u)=1.
"""
function cosine_series(coeffs::Vector{<:Real}; prec::Int=128)
    N = length(coeffs) - 1  # degree; coeffs[1] is DC, coeffs[k+1] is cos(ku)
    a_arb = [Arb(x; prec=prec) for x in coeffs]
    b_arb = [Arb(0; prec=prec) for _ in 0:N]
    return TrigPoly(a_arb, b_arb, prec)
end

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

"""
    tp_eval(f::TrigPoly, u::Arb) -> Arb

Evaluate f at the point u using rigorous Arb cos/sin arithmetic.
f(u) = a[1]/2 + Σ_{k=1}^{N} [a[k+1]*cos(ku) + b[k+1]*sin(ku)]
"""
function tp_eval(f::TrigPoly, u::Arb)
    p = f.prec
    result = Arb(f.a[1]; prec=p) / Arb(2; prec=p)
    N = degree(f)
    for k in 1:N
        ku = Arb(k; prec=p) * u
        result = result + f.a[k+1] * cos(ku) + f.b[k+1] * sin(ku)
    end
    return result
end

# Allow calling with a Float64 point (convert internally)
tp_eval(f::TrigPoly, u::Float64) = tp_eval(f, Arb(u; prec=f.prec))

# ---------------------------------------------------------------------------
# Hilbert transform
# ---------------------------------------------------------------------------

"""
    hilbert(f::TrigPoly) -> TrigPoly

Apply the periodic Hilbert transform:
  H[cos(ku)] = sin(ku)   for k >= 1
  H[sin(ku)] = -cos(ku)  for k >= 1
  H[1]       = 0
"""
function hilbert(f::TrigPoly)
    N, p = degree(f), f.prec
    a_out = Vector{Arb}(undef, N + 1)
    b_out = Vector{Arb}(undef, N + 1)
    # DC term: H[const] = 0
    a_out[1] = Arb(0; prec=p)
    b_out[1] = Arb(0; prec=p)
    for k in 1:N
        # H[a_k cos(ku) + b_k sin(ku)] = a_k sin(ku) - b_k cos(ku)
        # new cos coeff: -b_k
        # new sin coeff: +a_k
        a_out[k+1] = -f.b[k+1]
        b_out[k+1] = deepcopy(f.a[k+1])
    end
    return TrigPoly(a_out, b_out, p)
end

# ---------------------------------------------------------------------------
# Derivative
# ---------------------------------------------------------------------------

"""
    deriv(f::TrigPoly) -> TrigPoly

Differentiate: d/du[cos(ku)] = -k*sin(ku), d/du[sin(ku)] = k*cos(ku).
  (∂f)_a[k] = k * f.b[k]    (cos coefficient from differentiating sin terms)
  (∂f)_b[k] = -k * f.a[k]   (sin coefficient from differentiating cos terms)
"""
function deriv(f::TrigPoly)
    N, p = degree(f), f.prec
    a_out = [Arb(0; prec=p) for _ in 0:N]
    b_out = [Arb(0; prec=p) for _ in 0:N]
    # k=0: derivative of constant = 0; already zero
    for k in 1:N
        k_arb = Arb(k; prec=p)
        # d/du [b_k sin(ku)] = k*b_k*cos(ku) → contributes to a_out[k+1]
        a_out[k+1] = k_arb * f.b[k+1]
        # d/du [a_k cos(ku)] = -k*a_k*sin(ku) → contributes to b_out[k+1]
        b_out[k+1] = -(k_arb * f.a[k+1])
    end
    return TrigPoly(a_out, b_out, p)
end

# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------

"""
    add_const(f::TrigPoly, c::Real) -> TrigPoly

Return f + c (adds c to the DC term a[1], which represents a[1]/2 + c
after evaluation). We store 2c in a[1] so that evaluation gives (a[1]/2) = c.
"""
function add_const(f::TrigPoly, c::Real)
    p = f.prec
    a_out = copy(f.a)
    a_out[1] = f.a[1] + Arb(2 * c; prec=p)   # because eval uses a[1]/2
    return TrigPoly(a_out, copy(f.b), p)
end

function add_const(f::TrigPoly, c::Arb)
    p = f.prec
    a_out = copy(f.a)
    a_out[1] = f.a[1] + Arb(2; prec=p) * c
    return TrigPoly(a_out, copy(f.b), p)
end

# ---------------------------------------------------------------------------
# Addition and subtraction
# ---------------------------------------------------------------------------

function Base.:+(f::TrigPoly, g::TrigPoly)
    p = f.prec
    Nf, Ng = degree(f), degree(g)
    N = max(Nf, Ng)
    a_out = [Arb(0; prec=p) for _ in 0:N]
    b_out = [Arb(0; prec=p) for _ in 0:N]
    for k in 0:Nf
        a_out[k+1] = a_out[k+1] + f.a[k+1]
        b_out[k+1] = b_out[k+1] + f.b[k+1]
    end
    for k in 0:Ng
        a_out[k+1] = a_out[k+1] + g.a[k+1]
        b_out[k+1] = b_out[k+1] + g.b[k+1]
    end
    return TrigPoly(a_out, b_out, p)
end

function Base.:-(f::TrigPoly, g::TrigPoly)
    p = f.prec
    Nf, Ng = degree(f), degree(g)
    N = max(Nf, Ng)
    a_out = [Arb(0; prec=p) for _ in 0:N]
    b_out = [Arb(0; prec=p) for _ in 0:N]
    for k in 0:Nf
        a_out[k+1] = a_out[k+1] + f.a[k+1]
        b_out[k+1] = b_out[k+1] + f.b[k+1]
    end
    for k in 0:Ng
        a_out[k+1] = a_out[k+1] - g.a[k+1]
        b_out[k+1] = b_out[k+1] - g.b[k+1]
    end
    return TrigPoly(a_out, b_out, p)
end

function Base.:-(f::TrigPoly)
    p = f.prec
    N = degree(f)
    a_out = [-x for x in f.a]
    b_out = [-x for x in f.b]
    return TrigPoly(a_out, b_out, p)
end

# Scalar multiplication
function Base.:*(c::Arb, f::TrigPoly)
    p = f.prec
    a_out = [c * x for x in f.a]
    b_out = [c * x for x in f.b]
    return TrigPoly(a_out, b_out, p)
end

Base.:*(f::TrigPoly, c::Arb) = c * f
Base.:*(c::Real, f::TrigPoly) = Arb(c; prec=f.prec) * f
Base.:*(f::TrigPoly, c::Real) = c * f

# ---------------------------------------------------------------------------
# TrigPoly × TrigPoly via product-to-sum formulas
# ---------------------------------------------------------------------------

"""
    Base.:*(f::TrigPoly, g::TrigPoly) -> TrigPoly

Exact multiplication of two trig polynomials. Result has degree deg(f) + deg(g).

Uses the cos/sin product formulas:
  cos(ku) cos(ju) = (cos((k+j)u) + cos((k-j)u))/2
  sin(ku) sin(ju) = (cos((k-j)u) - cos((k+j)u))/2
  sin(ku) cos(ju) = (sin((k+j)u) + sin((k-j)u))/2
  cos(ku) sin(ju) = (sin((k+j)u) - sin((k-j)u))/2

Cost: O(deg(f) · deg(g)) Arb operations.
"""
function Base.:*(f::TrigPoly, g::TrigPoly)
    p = max(f.prec, g.prec)
    Nf = degree(f); Ng = degree(g); N = Nf + Ng

    # Normalised cos/sin coefficients: α_k = cos(ku) coefficient, β_k = sin(ku).
    # For k=0: α_0 is the actual DC value (= a[1]/2 in TrigPoly storage), β_0 = 0.
    half = Arb(1; prec=p) / Arb(2; prec=p)
    α_f(k) = k == 0 ? f.a[1] * half : f.a[k+1]
    β_f(k) = k == 0 ? Arb(0; prec=p) : f.b[k+1]
    α_g(k) = k == 0 ? g.a[1] * half : g.a[k+1]
    β_g(k) = k == 0 ? Arb(0; prec=p) : g.b[k+1]

    γ = [Arb(0; prec=p) for _ in 0:N]   # γ[n+1] = cos(nu) coeff of f·g
    δ = [Arb(0; prec=p) for _ in 0:N]   # δ[n+1] = sin(nu) coeff of f·g  (δ[1]=0)

    for k in 0:Nf, j in 0:Ng
        akαj = α_f(k) * α_g(j)
        akβj = α_f(k) * β_g(j)
        βkαj = β_f(k) * α_g(j)
        βkβj = β_f(k) * β_g(j)

        cos_pos = half * (akαj - βkβj)   # cos((k+j)u) coefficient contribution
        cos_neg = half * (akαj + βkβj)   # cos((k-j)u) coefficient contribution
        sin_pos = half * (βkαj + akβj)   # sin((k+j)u) coefficient contribution
        sin_neg = half * (βkαj - akβj)   # sin((k-j)u) coefficient contribution

        # k+j mode (always non-negative)
        kpj = k + j
        γ[kpj + 1] = γ[kpj + 1] + cos_pos
        δ[kpj + 1] = δ[kpj + 1] + sin_pos

        # |k-j| mode; handle k>j, k<j, k==j separately because of sin parity
        if k > j
            kmj = k - j
            γ[kmj + 1] = γ[kmj + 1] + cos_neg
            δ[kmj + 1] = δ[kmj + 1] + sin_neg
        elseif k < j
            kmj = j - k
            γ[kmj + 1] = γ[kmj + 1] + cos_neg          # cos is even
            δ[kmj + 1] = δ[kmj + 1] - sin_neg          # sin is odd: sin(k-j) = -sin(j-k)
        else  # k == j: |k-j| = 0, only cos contributes (sin(0) = 0)
            γ[1] = γ[1] + cos_neg
        end
    end

    # Convert (γ, δ) back to TrigPoly internal storage (a, b):
    #   a[1] = 2 · γ[1]  (so a[1]/2 = DC = γ[0])
    #   a[k+1] = γ[k+1], b[k+1] = δ[k+1]  for k ≥ 1
    a_out = Vector{Arb}(undef, N + 1)
    b_out = Vector{Arb}(undef, N + 1)
    a_out[1] = Arb(2; prec=p) * γ[1]
    b_out[1] = Arb(0; prec=p)
    for k in 1:N
        a_out[k+1] = γ[k+1]
        b_out[k+1] = δ[k+1]
    end
    return TrigPoly(a_out, b_out, p)
end

# ---------------------------------------------------------------------------
# Sup-norm bound
# ---------------------------------------------------------------------------

"""
    supnorm_bound(f::TrigPoly) -> Arb

Return an upper bound on ‖f‖_∞ via the triangle-inequality estimate:
  ‖f‖_∞ ≤ |a[1]|/2 + Σ_{k=1}^{N} (|a[k+1]| + |b[k+1]|)
"""
function supnorm_bound(f::TrigPoly)
    p = f.prec
    s = abs(f.a[1]) / Arb(2; prec=p)
    N = degree(f)
    for k in 1:N
        s = s + abs(f.a[k+1]) + abs(f.b[k+1])
    end
    return s
end

# ---------------------------------------------------------------------------
# Equality check (for testing)
# ---------------------------------------------------------------------------

"""
    trigpoly_equal(f::TrigPoly, g::TrigPoly) -> Bool

Check whether two TrigPolys have overlapping (i.e., consistent) coefficients.
Pads the shorter one with zeros and checks that each pair of coefficient balls overlaps.
"""
function trigpoly_equal(f::TrigPoly, g::TrigPoly)
    Nf, Ng = degree(f), degree(g)
    N = max(Nf, Ng)
    p = f.prec

    zero_arb = Arb(0; prec=p)

    for k in 0:N
        fa = k <= Nf ? f.a[k+1] : zero_arb
        ga = k <= Ng ? g.a[k+1] : zero_arb
        fb = k <= Nf ? f.b[k+1] : zero_arb
        gb = k <= Ng ? g.b[k+1] : zero_arb

        # Check that fa - ga contains zero (i.e., balls overlap)
        Arblib.contains_zero(fa - ga) || return false
        Arblib.contains_zero(fb - gb) || return false
    end
    return true
end

# ---------------------------------------------------------------------------
# Exports (for use from the parent module)
# ---------------------------------------------------------------------------

export TrigPoly, degree, hilbert, deriv, tp_eval, supnorm_bound
export cosine_series, add_const, trigpoly_equal

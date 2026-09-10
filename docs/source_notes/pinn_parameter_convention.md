# PINN parameter convention — `lamb`, `beta` ≠ Babenko `μ`, `β`

**Source:** Michal Shavit's training notebook `~/My Drive/PINN_2.ipynb`,
and the Wilton-ripple replay copy `~/Downloads/6_Wilton_ripple_High.ipynb`.
**Action item for whoever writes `BabenkoRigorous/src/Certificate.jl`'s
emitter / runs `residual_bound` on a PINN solution:** the PINN's saved
`lamb` and `beta` are **not** the Babenko `μ` and `β` used in
`BabenkoRigorous/src/ResidualBound.jl` and `WaveCAP/.../Definitions.lean`.

## 1. The PINN's equation form

In `PINN_2.ipynb` cell `stwv_gov_eqn` (line ~246):

```python
e1 = (
    -lamb**2 * Hy_x
     + g * y
     + g * 0.5 * Hysq_x          # = g · H[y·y_x],  since (y²)_x = 2·y·y_x
     + g * y * Hy_x
     + (beta)**2 * (-dtens1 + Hdtens2)
)
```

Here

- `Hy_x   = H[y_x]`
- `Hysq_x = H[(y²)_x] = 2·H[y·y_x]`
- `dtens1 = ∂_x(y_x / D)`        (from the relation `t1 = y_x / D`)
- `Hdtens2 = H[∂_x((1+H[y_x])/D)]` (from `t2 = (1+H[y_x])/D`)
- `g = 1.0` is set in the notebook.

So with `g = 1`, the PINN trains `y` to satisfy

```
−λ²·H[y_x] + y + H[y·y_x] + y·H[y_x] + β²·(−cap1 + cap2) = 0
```

i.e. equivalently

```
λ²·H[y_x] = y + y·H[y_x] + H[y·y_x] − β²·cap1 + β²·cap2.
```

## 2. The `BabenkoRigorous` / `WaveCAP` form

From `docs/overview.md` and the Julia/Lean residual definitions:

```
μ·H[y_u] = y + y·H[y_u] + H(y·y_u) − β·∂_u(y_u/D) + β·H[∂_u((1+H[y_u])/D)]
```

This is identical to the PINN's equation **after the substitution**

```
μ_Babenko = (lamb_PINN)²
β_Babenko = (beta_PINN)²
```

(with `g = 1`).

## 3. Worked example for the shipped weights

The file `StokesWave_lamd=1.1688_beta=0.4980_g=1.0_weights.pkl` records

```
lamb_PINN = 1.168786478294966
beta_PINN = 0.498000000104982
```

Hence

```
μ_Babenko = (1.168786478294966)² = 1.366061831845…
β_Babenko = (0.498000000104982)² = 0.248004000104…
```

Verification: the Float64 residual of the projected coefficients
(`pinn_a_coeffs_N110.txt`) evaluated under the `BabenkoRigorous` form with
these `μ_Babenko, β_Babenko` is

```
‖R‖_∞ ≈ 1.518 × 10⁻⁷         (FFT, N = 8192)
```

— consistent with a well-trained PINN. Using `lamb` and `beta` directly
(without squaring) gives `‖R‖_∞ ≈ 1.87`, off by seven orders of magnitude.

This sanity check is built into `BabenkoRigorous/scripts/run_pinn.jl`.

## 4. What the certificate emitter must do

When `BabenkoRigorous/src/Certificate.jl` produces a `Certificate.lean`
for a PINN-trained solution, it must:

1. Read `lamb_PINN, beta_PINN` (e.g. from the pickle filename or a
   companion metadata file).
2. Write into `Certificate.lean`:
   ```lean
   noncomputable def μ_cert : ℝ := /- lamb_PINN^2 -/
   noncomputable def β_cert : ℝ := /- beta_PINN^2 -/
   ```
   — **not** the raw PINN values.
3. Make sure the same `μ, β` get passed to `residual_bound(a_coeffs, μ, β; …)`
   when computing `ε₀_cert`. Otherwise the Lean trust anchor
   `cert_residual_le : ∀ u, |R a_cert μ_cert β_cert u| ≤ ε₀_cert` would be
   asserting the bound at *different* parameters than the Julia computation
   actually certifies.

## 5. Cross-references

- PINN training notebook: `~/My Drive/PINN_2.ipynb` cell `stwv_gov_eqn`
  (also `~/Downloads/6_Wilton_ripple_High.ipynb`).
- Weights file (public): `https://raw.githubusercontent.com/michalshavitNYU/Waves/main/StokesWave_lamd%3D1.1688_beta%3D0.4980_g%3D1.0_weights.pkl`.
- Babenko equation form on the WaveCAP side: `docs/overview.md` §1, and
  `WaveCAP/WaveCAP/Definitions.lean` (`R` definition).
- Verification harness: `BabenkoRigorous/scripts/run_pinn.jl`.

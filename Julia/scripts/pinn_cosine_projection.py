"""
pinn_cosine_projection.py — Project the PINN-trained Stokes wave onto a cosine
series via composite 4-point Gauss-Legendre quadrature and plot truncation error.

Source notebook: ~/Downloads/Plot_PINN_Water_Waves.ipynb
Weights URL: https://raw.githubusercontent.com/michalshavitNYU/Waves/main/
             StokesWave_lamd=1.1688_beta=0.4980_g=1.0_weights.pkl

The PINN evaluates y(x) = scl * NN(cos x), so y is even in x and admits a pure
cosine series

    y(x) = a_0/2 + sum_{k>=1} a_k cos(k x),     a_k = (2/pi) int_0^pi y(x) cos(kx) dx.

We compute each a_k by composite 4-point Gauss-Legendre on [0, pi] with M panels.
The 4-point rule integrates polynomials of degree <= 7 exactly per panel, so with
analytic y the total error decays like h^8.
"""
import os
import pickle
import urllib.request

import jax
jax.config.update("jax_enable_x64", True)  # MUST be set before any jnp ops
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# PINN definition (copied from the notebook)
# ---------------------------------------------------------------------------
WEIGHTS_URL = (
    "https://raw.githubusercontent.com/michalshavitNYU/Waves/main/"
    "StokesWave_lamd%3D1.1688_beta%3D0.4980_g%3D1.0_weights.pkl"
)
CACHE_PATH = os.path.join(os.path.dirname(__file__), "_pinn_weights.pkl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

C    = 1.168786478294966
BETA = 0.498000000104982
SCL  = 0.27


def download_weights():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    print(f"Downloading weights from {WEIGHTS_URL}")
    raw = urllib.request.urlopen(WEIGHTS_URL).read()
    with open(CACHE_PATH, "wb") as f:
        f.write(raw)
    return pickle.loads(raw)


def to_float64(params):
    """Recursively cast a pytree of arrays to float64.  The pickled weights
    are float32; running them at float64 removes the ~1e-6 single-precision
    forward-pass noise floor and lets us see the true PINN accuracy."""
    out = []
    for layer in params:
        W, b = layer
        out.append((np.asarray(W, dtype=np.float64),
                    np.asarray(b, dtype=np.float64)))
    return out


def network_core(params, H):
    first, *hidden, last = params
    H = jnp.tanh(jnp.dot(H, first[0]) + first[1])
    for layer in hidden:
        H = jnp.tanh(jnp.dot(H, layer[0]) + layer[1])
    return jnp.dot(H, last[0]) + last[1]


def y_of_x(x: np.ndarray, net_params) -> np.ndarray:
    """Vectorised evaluation y(x) = SCL * NN(cos x).  Accepts 1D numpy x.
    Assumes net_params have been cast to float64 (see to_float64)."""
    H = np.cos(np.asarray(x, dtype=np.float64))[:, None]
    out = SCL * np.array(network_core(net_params, jnp.asarray(H)))
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Composite 4-point Gauss-Legendre quadrature on [0, pi]
# ---------------------------------------------------------------------------
# Nodes and weights for 4-point Gauss-Legendre on [-1, 1].
_GL4_NODES, _GL4_WEIGHTS = np.polynomial.legendre.leggauss(4)


def gl4_grid(M_panels: int, a: float = 0.0, b: float = np.pi):
    """Composite 4-point Gauss-Legendre nodes & weights on [a, b]."""
    edges = np.linspace(a, b, M_panels + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])           # (M,)
    halfh   = 0.5 * (edges[1:] - edges[:-1])           # (M,)
    nodes   = centers[:, None] + halfh[:, None] * _GL4_NODES[None, :]   # (M, 4)
    weights = halfh[:, None] * _GL4_WEIGHTS[None, :]                    # (M, 4)
    return nodes.reshape(-1), weights.reshape(-1)


def cosine_coefficients(y_vals: np.ndarray, x_nodes: np.ndarray,
                        weights: np.ndarray, N: int) -> np.ndarray:
    """Return a[0..N] with a_k = (2/pi) int_0^pi y(x) cos(kx) dx via the supplied
    quadrature rule.  Uses the convention y(x) = a_0/2 + sum_{k>=1} a_k cos(kx)."""
    a = np.empty(N + 1)
    wy = weights * y_vals          # (Q,)
    for k in range(N + 1):
        a[k] = (2.0 / np.pi) * np.sum(wy * np.cos(k * x_nodes))
    return a


def evaluate_cosine_series(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate y_N(x) = a_0/2 + sum_{k=1}^N a_k cos(kx)."""
    y = 0.5 * a[0] * np.ones_like(x)
    for k in range(1, len(a)):
        y += a[k] * np.cos(k * x)
    return y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== PINN cosine-series projection ===")
    print(f"  c    = {C}")
    print(f"  beta = {BETA}")
    print(f"  scl  = {SCL}")

    trained = download_weights()
    net_params = to_float64(trained[0])
    print("  network params cast to float64 (originally float32).")

    # ---- sanity: reproduce the notebook's plot ----
    x_plot = np.linspace(-np.pi, np.pi, 4096, endpoint=False)
    y_plot = y_of_x(x_plot, net_params)
    print(f"  y range:        [{y_plot.min():+.9f}, {y_plot.max():+.9f}]")
    print(f"  y mean:         {y_plot.mean():+.9e}")
    print(f"  y(0), y(pi):    {y_plot[len(x_plot)//2]:+.9e}, {y_plot[-1]:+.9e}")

    # ---- panel-count sweep to confirm convergence ----
    print("\n--- Panel-count sweep at N_MAX = 512 ---")
    N_MAX_SWEEP = 512
    M_list = [256, 1024, 4096, 16384]
    a_prev = None
    a_sweep = {}
    for M in M_list:
        x_q, w_q = gl4_grid(M)
        y_q = y_of_x(x_q, net_params)
        a_M = cosine_coefficients(y_q, x_q, w_q, N_MAX_SWEEP)
        a_sweep[M] = a_M
        if a_prev is not None:
            diff = np.max(np.abs(a_M - a_prev))
            print(f"  M={M:6d} ({4*M:6d} nodes):  "
                  f"max|a_k(M)-a_k(M_prev)| = {diff:.3e}")
        else:
            print(f"  M={M:6d} ({4*M:6d} nodes):  (baseline)")
        a_prev = a_M

    # ---- pick the finest quadrature for downstream use ----
    M_PANELS = M_list[-1]
    N_MAX    = N_MAX_SWEEP
    a = a_sweep[M_PANELS]
    print(f"\nUsing finest quadrature: M={M_PANELS} panels = {4*M_PANELS} nodes.")

    print("\nFirst 10 cosine coefficients:")
    for k in range(10):
        print(f"  a[{k:3d}] = {a[k]:+.12e}")

    print(f"\n|a_k| at the tail:")
    for k in (32, 64, 128, 192, 256, 384, 512):
        if k < len(a):
            print(f"  |a[{k:3d}]| = {abs(a[k]):.3e}")

    # ---- truncation error: y_N(x) vs y(x) on a dense plot grid ----
    Ns = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
    x_dense = np.linspace(0.0, np.pi, 8193)
    y_dense = y_of_x(x_dense, net_params)

    # ---- Figure 1: PINN solution itself ----
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(x_plot, y_plot, lw=1.2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y(x)")
    ax1.set_title(f"PINN Stokes wave  (lambda=1.1688, beta={BETA:.4f}, scl={SCL})")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUT_DIR, "pinn_y_plot.png"), dpi=140)

    # ---- Figure 2: cosine coefficient magnitudes (semilogy) ----
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ks = np.arange(N_MAX + 1)
    ax2.semilogy(ks, np.abs(a) + 1e-300, marker=".", lw=0.8)
    ax2.set_xlabel("mode k")
    ax2.set_ylabel("|a_k|")
    ax2.set_title("Cosine coefficients of PINN solution (4-pt GL quadrature)")
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "pinn_coeffs.png"), dpi=140)

    # ---- Figure 3: pointwise truncation error vs x for several N ----
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    sup_errs = []
    for N in Ns:
        y_N = evaluate_cosine_series(a[: N + 1], x_dense)
        err = np.abs(y_dense - y_N)
        sup_errs.append(err.max())
        ax3.semilogy(x_dense, err + 1e-300, label=f"N={N}  (sup={err.max():.2e})", lw=0.9)
    ax3.set_xlabel("x")
    ax3.set_ylabel("|y(x) - y_N(x)|")
    ax3.set_title("Cosine-series truncation error")
    ax3.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["0", "pi/4", "pi/2", "3pi/4", "pi"])
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUT_DIR, "pinn_truncation_error.png"), dpi=140)

    # ---- Figure 4: sup-norm truncation error vs N ----
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.semilogy(Ns, sup_errs, marker="o")
    ax4.set_xlabel("truncation order N")
    ax4.set_ylabel("max_x |y - y_N|")
    ax4.set_title("Sup-norm of cosine-series truncation error")
    ax4.grid(True, which="both", alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUT_DIR, "pinn_sup_error_vs_N.png"), dpi=140)

    print("\nSup-norm truncation errors:")
    for N, e in zip(Ns, sup_errs):
        print(f"  N={N:4d}:   max|y - y_N| = {e:.3e}")

    # ---- save coefficients for downstream BabenkoRigorous use ----
    coeff_path = os.path.join(OUT_DIR, "..", "pinn_cosine_coeffs.npz")
    np.savez(coeff_path, a=a, c=C, beta=BETA, scl=SCL,
             M_panels=M_PANELS, N_max=N_MAX)
    print(f"\nSaved cosine coefficients to {coeff_path}")
    print(f"Figures saved under {OUT_DIR}/pinn_*.png")


if __name__ == "__main__":
    main()

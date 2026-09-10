"""
plot_cosine_approx.py — Visualise the N=110 cosine approximation of the PINN
Stokes wave alongside the PINN evaluation, and the pointwise error.

Reads coefficients from BabenkoRigorous/pinn_a_coeffs_N110.txt, re-evaluates
y(x) via the network for ground truth, reconstructs y_N(x) from the cosine
series, and saves a two-panel figure.
"""
import os
import pickle
import urllib.request

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(__file__)
COEFF_PATH = os.path.join(THIS_DIR, "..", "pinn_a_coeffs_N110.txt")
CACHE_PATH = os.path.join(THIS_DIR, "_pinn_weights.pkl")
OUT_DIR = os.path.join(THIS_DIR, "..", "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

WEIGHTS_URL = (
    "https://raw.githubusercontent.com/michalshavitNYU/Waves/main/"
    "StokesWave_lamd%3D1.1688_beta%3D0.4980_g%3D1.0_weights.pkl"
)
SCL = 0.27


def load_coeffs(path):
    a = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            a.append(float(s))
    return np.array(a)


def network_core(params, H):
    first, *hidden, last = params
    H = jnp.tanh(jnp.dot(H, first[0]) + first[1])
    for layer in hidden:
        H = jnp.tanh(jnp.dot(H, layer[0]) + layer[1])
    return jnp.dot(H, last[0]) + last[1]


def to_float64(params):
    return [(np.asarray(W, dtype=np.float64), np.asarray(b, dtype=np.float64))
            for W, b in params]


def y_pinn(x, net_params):
    H = np.cos(np.asarray(x, dtype=np.float64))[:, None]
    return SCL * np.asarray(network_core(net_params, jnp.asarray(H))).reshape(-1)


def y_cosine(a, x):
    y = 0.5 * a[0] * np.ones_like(x)
    for k in range(1, len(a)):
        y += a[k] * np.cos(k * x)
    return y


def main():
    # Load coefficients and network weights.
    a = load_coeffs(COEFF_PATH)
    N = len(a) - 1
    print(f"Loaded {len(a)} coefficients (degree N = {N}).")

    if not os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "wb") as f:
            f.write(urllib.request.urlopen(WEIGHTS_URL).read())
    with open(CACHE_PATH, "rb") as f:
        net_params = to_float64(pickle.load(f)[0])

    # Evaluate both on a dense grid spanning [-pi, pi].
    x = np.linspace(-np.pi, np.pi, 4097)
    y_true = y_pinn(x, net_params)
    y_N    = y_cosine(a, x)
    err    = y_true - y_N
    print(f"sup |y_PINN - y_N|  = {np.max(np.abs(err)):.3e}")

    # ---- composite figure ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                   gridspec_kw={"height_ratios": [2, 1]},
                                   sharex=True)

    ax1.plot(x, y_true, lw=2.0, label="PINN  y(x) = 0.27·NN(cos x)", color="C0")
    ax1.plot(x, y_N, lw=1.0, ls="--", color="C3",
             label=f"cosine reconstruction y_N(x), N={N}")
    ax1.set_ylabel("y(x)")
    ax1.set_title(f"PINN Stokes wave vs N={N} cosine reconstruction"
                  f"  (4-pt GL projection, M=16384 panels)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower center")

    ax2.semilogy(x, np.abs(err) + 1e-300, lw=0.8, color="C2")
    ax2.set_xlabel("x")
    ax2.set_ylabel("|y_PINN − y_N|")
    ax2.set_title(f"Pointwise reconstruction error"
                  f"  (sup = {np.max(np.abs(err)):.2e})")
    ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                   ["−π", "−π/2", "0", "π/2", "π"])
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "pinn_cosine_approx.png")
    fig.savefig(out, dpi=140)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()

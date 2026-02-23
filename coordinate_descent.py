"""
Iteration 1 & 2 + Control — Coordinate Descent Experiments
============================================================
Implements:
  1. Logistic loss, full gradient, coordinate gradient, coordinate Lipschitz
  2. Gauss–Southwell (GS) coordinate descent
  3. Normalized GS coordinate descent
  4. Random coordinate descent (control)
  5. Convergence plot comparing all three against L*
"""

import numpy as np
import os
import time

# ──────────────────────────────────────────────
# 1. Load data and L*
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

X = np.loadtxt(os.path.join(BASE_DIR, "X.tsv"), delimiter="\t", skiprows=1)
y = np.loadtxt(os.path.join(BASE_DIR, "y.tsv"), delimiter="\t", skiprows=1).astype(int)

with open(os.path.join(BASE_DIR, "L_star.txt")) as f:
    L_star = float(f.read().strip())

n, d = X.shape
print(f"Data: {n} samples, {d} features")
print(f"L* = {L_star:.10e}")

# ──────────────────────────────────────────────
# 2. Core functions
# ──────────────────────────────────────────────

def sigmoid(t):
    """Numerically stable sigmoid."""
    return np.where(t >= 0,
                    1.0 / (1.0 + np.exp(-t)),
                    np.exp(t) / (1.0 + np.exp(t)))


def logistic_loss(w, X, y):
    """L(w) = sum_n log(1 + exp(-y_n * x_n^T w))"""
    z = y * (X @ w)
    return np.sum(np.logaddexp(0, -z))


def full_gradient(w, X, y):
    """∂L/∂w = -X^T (y ⊙ (1 - σ(y ⊙ Xw)))"""
    z = y * (X @ w)
    s = sigmoid(z)  # σ(y_n * x_n^T w)
    # gradient_n = -y_n * x_n * (1 - σ(z_n))
    coeff = -y * (1 - s)  # shape (n,)
    return X.T @ coeff    # shape (d,)


def coordinate_gradient(w, X, y, i, s=None):
    """∂L/∂w_i  (scalar)"""
    if s is None:
        z = y * (X @ w)
        s = sigmoid(z)
    coeff = -y * (1 - s)
    return X[:, i] @ coeff


def coordinate_lipschitz(X, y, i, s):
    """
    L_i = sum_n x_{n,i}^2 * σ(z_n) * (1 - σ(z_n))
    where σ(z_n) = sigmoid(y_n * x_n^T w)
    """
    return np.sum(X[:, i] ** 2 * s * (1 - s))


# ──────────────────────────────────────────────
# 3. Coordinate descent solvers
# ──────────────────────────────────────────────

def coordinate_descent(X, y, method="gs", max_iter=5000, seed=42):
    """
    Run coordinate descent on logistic loss.

    method: 'gs'            — Gauss–Southwell (argmax |∂L/∂w_i|)
            'normalized_gs' — Normalized GS (argmax |∂L/∂w_i| / sqrt(L_i))
            'random'        — Uniform random coordinate
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)

    losses = [logistic_loss(w, X, y)]
    t_start = time.time()

    for t in range(max_iter):
        # Compute sigmoid once per iteration (shared by gradient & Lipschitz)
        z = y * (X @ w)
        s = sigmoid(z)

        if method == "random":
            i = rng.integers(d)
        else:
            # Compute all coordinate gradients and Lipschitz constants
            grad = full_gradient(w, X, y)  # reuse via precomputed s — but let's use the s we have
            # Actually recompute using our s for consistency:
            coeff = -y * (1 - s)
            grad = X.T @ coeff

            if method == "gs":
                i = np.argmax(np.abs(grad))
            elif method == "normalized_gs":
                Li = np.array([coordinate_lipschitz(X, y, j, s) for j in range(d)])
                # Avoid division by zero
                Li = np.maximum(Li, 1e-12)
                scores = np.abs(grad) / np.sqrt(Li)
                i = np.argmax(scores)
            else:
                raise ValueError(f"Unknown method: {method}")

        # Coordinate gradient and Lipschitz constant for selected i
        g_i = coordinate_gradient(w, X, y, i, s=s)
        L_i = coordinate_lipschitz(X, y, i, s)

        # Avoid zero Lipschitz (happens when predictions are very confident)
        if L_i < 1e-12:
            L_i = 1e-12

        # Update
        w[i] -= g_i / L_i

        losses.append(logistic_loss(w, X, y))

    elapsed = time.time() - t_start
    return w, np.array(losses), elapsed


# ──────────────────────────────────────────────
# 4. Run experiments
# ──────────────────────────────────────────────
MAX_ITER = 5000

print(f"\nRunning {MAX_ITER} iterations for each method...\n")

w_gs, losses_gs, time_gs = coordinate_descent(X, y, method="gs", max_iter=MAX_ITER)
print(f"GS CD:            final loss = {losses_gs[-1]:.6f},  time = {time_gs:.2f}s")

w_ngs, losses_ngs, time_ngs = coordinate_descent(X, y, method="normalized_gs", max_iter=MAX_ITER)
print(f"Normalized GS CD: final loss = {losses_ngs[-1]:.6f},  time = {time_ngs:.2f}s")

# Run random CD 5 times with different seeds for error bars
RANDOM_SEEDS = [0, 1, 2, 3, 4]
random_runs = []
for seed in RANDOM_SEEDS:
    w_r, losses_r, time_r = coordinate_descent(X, y, method="random", max_iter=MAX_ITER, seed=seed)
    random_runs.append(losses_r)
    print(f"Random CD (seed={seed}): final loss = {losses_r[-1]:.6f},  time = {time_r:.2f}s")

random_all = np.array(random_runs)          # shape (5, MAX_ITER+1)
losses_rand_mean = random_all.mean(axis=0)
losses_rand_std  = random_all.std(axis=0)

# ──────────────────────────────────────────────
# 5. Summary
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Method':<22} {'Final Loss':>12} {'Time (s)':>10} {'L*':>14}")
print(f"{'='*60}")
print(f"{'GS':<22} {losses_gs[-1]:>12.6f} {time_gs:>10.2f} {L_star:>14.6e}")
print(f"{'Normalized GS':<22} {losses_ngs[-1]:>12.6f} {time_ngs:>10.2f} {L_star:>14.6e}")
print(f"{'Random (mean±std)':<22} {losses_rand_mean[-1]:>12.6f} {'':>10} {L_star:>14.6e}")
print(f"{'='*60}")

# ──────────────────────────────────────────────
# 6. Save loss histories
# ──────────────────────────────────────────────
iters = np.arange(len(losses_gs))
loss_data = np.column_stack([iters, losses_gs, losses_ngs, losses_rand_mean, losses_rand_std])
np.savetxt(
    os.path.join(BASE_DIR, "cd_losses.tsv"),
    loss_data,
    delimiter="\t",
    header="iteration\tGS\tNormalized_GS\tRandom_mean\tRandom_std",
    comments="",
    fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"],
)
print(f"\nSaved cd_losses.tsv ({len(losses_gs)} rows)")

# ──────────────────────────────────────────────
# 7. Convergence plot
# ──────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(losses_gs, label="GS Coordinate Descent", linewidth=1.5)
    ax.plot(losses_ngs, label="Normalized GS Coordinate Descent", linewidth=1.5)

    # Random CD: mean line with ±1 std shaded region
    iters_arr = np.arange(len(losses_rand_mean))
    ax.plot(iters_arr, losses_rand_mean, label="Random Coordinate Descent (mean)", linewidth=1.5, alpha=0.8)
    ax.fill_between(
        iters_arr,
        np.maximum(losses_rand_mean - losses_rand_std, 1e-15),
        losses_rand_mean + losses_rand_std,
        alpha=0.25, label="Random CD ±1 std (5 runs)",
    )
    ax.axhline(y=L_star, color="black", linestyle="--", linewidth=1, label=f"L* = {L_star:.2e}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Logistic Loss")
    ax.set_title("Coordinate Descent Convergence Comparison")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(BASE_DIR, "convergence_plot.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")
    plt.close(fig)

except ImportError:
    print("matplotlib not available — skipping plot")

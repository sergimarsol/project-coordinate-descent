"""
Iteration 3 — Sparse Coordinate Descent
=========================================
Implements k-sparse coordinate descent using an active set.
Uses normalized GS rule for coordinate selection.
Evaluates loss for k = 1, 3, 5, 7, 10, 13.
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

FEATURE_NAMES = [
    "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols",
    "Proanthocyanins", "Color intensity", "Hue",
    "OD280/OD315", "Proline",
]

n, d = X.shape
print(f"Data: {n} samples, {d} features")
print(f"L* = {L_star:.10e}\n")

# ──────────────────────────────────────────────
# 2. Core functions (same as coordinate_descent.py)
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


def coordinate_lipschitz(X, y, i, s):
    """L_i = sum_n x_{n,i}^2 * σ(z_n) * (1 - σ(z_n))"""
    return np.sum(X[:, i] ** 2 * s * (1 - s))


# ──────────────────────────────────────────────
# 3. Sparse coordinate descent
# ──────────────────────────────────────────────

def sparse_coordinate_descent(X, y, k, max_iter=5000):
    """
    k-sparse coordinate descent with active set.

    Uses normalized Gauss–Southwell rule for selection.
    Maintains active set S with |S| <= k.
    """
    n, d = X.shape
    w = np.zeros(d)
    S = set()  # active set of non-zero coordinate indices

    losses = [logistic_loss(w, X, y)]

    for t in range(max_iter):
        # Compute sigmoid once
        z = y * (X @ w)
        s = sigmoid(z)

        # Compute all coordinate gradients
        coeff = -y * (1 - s)
        grad = X.T @ coeff

        # Compute all coordinate Lipschitz constants
        sv = s * (1 - s)  # shape (n,)
        Li = (X ** 2).T @ sv  # shape (d,), vectorized

        # Avoid division by zero
        Li_safe = np.maximum(Li, 1e-12)

        # Normalized GS scores
        scores = np.abs(grad) / np.sqrt(Li_safe)

        # Select coordinate with highest score
        i = np.argmax(scores)

        # Active set management
        if i in S:
            # Already active — update normally
            pass
        elif len(S) < k:
            # Room in active set — add it
            S.add(i)
        else:
            # Active set full — swap out the smallest weight
            j_remove = min(S, key=lambda j: abs(w[j]))
            w[j_remove] = 0.0
            S.discard(j_remove)
            S.add(i)

        # Coordinate update (only for active coordinates)
        g_i = grad[i]
        L_i = Li_safe[i]
        w[i] -= g_i / L_i

        # Enforce sparsity: zero out anything not in S
        mask = np.ones(d, dtype=bool)
        mask[list(S)] = False
        w[mask] = 0.0

        losses.append(logistic_loss(w, X, y))

    return w, np.array(losses), S


# ──────────────────────────────────────────────
# 4. Run experiments for different k
# ──────────────────────────────────────────────
K_VALUES = [1, 3, 5, 7, 10, 13]
MAX_ITER = 5000

results = {}

print(f"{'k':>3}  {'Final Loss':>14}  {'||w||_0':>7}  Active Features")
print("-" * 70)

for k in K_VALUES:
    t0 = time.time()
    w_k, losses_k, S_k = sparse_coordinate_descent(X, y, k, max_iter=MAX_ITER)
    elapsed = time.time() - t0

    nnz = np.count_nonzero(w_k)
    active_names = [f"{FEATURE_NAMES[j]}" for j in sorted(S_k)]

    results[k] = {
        "w": w_k,
        "losses": losses_k,
        "S": S_k,
        "time": elapsed,
    }

    print(f"{k:>3}  {losses_k[-1]:>14.6f}  {nnz:>7}  {', '.join(active_names)}")

# ──────────────────────────────────────────────
# 5. Summary table
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SPARSE CD RESULTS (Normalized GS, {MAX_ITER} iterations)")
print(f"{'='*60}")
print(f"{'k':>3}  {'Final Loss':>14}  {'Time (s)':>10}  {'Gap to L*':>14}")
print(f"{'-'*60}")
for k in K_VALUES:
    fl = results[k]["losses"][-1]
    gap = fl - L_star
    print(f"{k:>3}  {fl:>14.6f}  {results[k]['time']:>10.2f}  {gap:>14.6e}")
print(f"{'L*':>3}  {L_star:>14.6e}")
print(f"{'='*60}")

# ──────────────────────────────────────────────
# 6. Save loss table
# ──────────────────────────────────────────────
table_lines = ["k\tFinal_Loss\tGap_to_Lstar\tActive_Features"]
for k in K_VALUES:
    fl = results[k]["losses"][-1]
    gap = fl - L_star
    active = ";".join(FEATURE_NAMES[j] for j in sorted(results[k]["S"]))
    table_lines.append(f"{k}\t{fl:.8f}\t{gap:.8e}\t{active}")

with open(os.path.join(BASE_DIR, "sparse_results.tsv"), "w") as f:
    f.write("\n".join(table_lines) + "\n")
print(f"\nSaved sparse_results.tsv")

# Save loss curves for each k
iters = np.arange(MAX_ITER + 1)
cols = [iters]
headers = ["iteration"]
for k in K_VALUES:
    cols.append(results[k]["losses"])
    headers.append(f"k={k}")

loss_data = np.column_stack(cols)
np.savetxt(
    os.path.join(BASE_DIR, "sparse_losses.tsv"),
    loss_data,
    delimiter="\t",
    header="\t".join(headers),
    comments="",
    fmt=["%d"] + ["%.8f"] * len(K_VALUES),
)
print(f"Saved sparse_losses.tsv")

# ──────────────────────────────────────────────
# 7. Convergence plot
# ──────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for k in K_VALUES:
        ax.plot(results[k]["losses"], label=f"k = {k}", linewidth=1.5)

    ax.axhline(y=L_star, color="black", linestyle="--", linewidth=1,
               label=f"L* = {L_star:.2e}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Logistic Loss")
    ax.set_title("Sparse Coordinate Descent — Loss vs k")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(BASE_DIR, "sparse_convergence_plot.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")
    plt.close(fig)

    # Bar chart of final loss vs k
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    final_losses = [results[k]["losses"][-1] for k in K_VALUES]
    bars = ax2.bar([str(k) for k in K_VALUES], final_losses, color="steelblue")
    ax2.axhline(y=L_star, color="red", linestyle="--", linewidth=1,
                label=f"L* = {L_star:.2e}")
    ax2.set_xlabel("k (sparsity)")
    ax2.set_ylabel("Final Logistic Loss")
    ax2.set_title("Final Loss vs Sparsity Level k")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig2.tight_layout()
    bar_path = os.path.join(BASE_DIR, "sparse_loss_bar.png")
    fig2.savefig(bar_path, dpi=150)
    print(f"Saved {bar_path}")
    plt.close(fig2)

except ImportError:
    print("matplotlib not available — skipping plots")

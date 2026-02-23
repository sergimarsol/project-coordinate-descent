"""
Step 1 — Baseline Optimal Loss (L*)
=====================================
Train unregularized logistic regression using sklearn (C = 1e10)
and compute the logistic loss at the learned weights.

L* serves as the convergence target for our coordinate descent methods.
"""

import numpy as np
import os
from sklearn.linear_model import LogisticRegression

# ──────────────────────────────────────────────
# 1. Load preprocessed data
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

X = np.loadtxt(os.path.join(BASE_DIR, "X.tsv"), delimiter="\t", skiprows=1)
y = np.loadtxt(os.path.join(BASE_DIR, "y.tsv"), delimiter="\t", skiprows=1).astype(int)

print(f"Loaded X: {X.shape}, y: {y.shape}")
print(f"Labels: +1 -> {np.sum(y == 1)}, -1 -> {np.sum(y == -1)}")

# ──────────────────────────────────────────────
# 2. Logistic loss function
# ──────────────────────────────────────────────
def logistic_loss(w, X, y):
    """
    L(w) = sum_n log(1 + exp(-y_n * (w^T x_n)))

    Uses the log-sum-exp trick for numerical stability:
        log(1 + exp(t)) = max(0, t) + log(1 + exp(-|t|))
    """
    z = y * (X @ w)
    return np.sum(np.logaddexp(0, -z))

# ──────────────────────────────────────────────
# 3. Train sklearn logistic regression (effectively unregularized)
# ──────────────────────────────────────────────
# sklearn doesn't support truly unregularized LR, so we set C very large.
# fit_intercept=False because the roadmap formulation has no bias term,
# and our features are already standardized (zero mean).
model = LogisticRegression(
    C=1e10,
    solver="lbfgs",
    fit_intercept=False,
    max_iter=10000,
    tol=1e-12,
)
model.fit(X, y)

# ──────────────────────────────────────────────
# 4. Extract learned weights and compute L*
# ──────────────────────────────────────────────
w_star = model.coef_.flatten()
L_star = logistic_loss(w_star, X, y)

print(f"\n{'='*50}")
print(f"sklearn Logistic Regression (C=1e10)")
print(f"{'='*50}")
print(f"Learned weights w*:")
for i, wi in enumerate(w_star):
    print(f"  w[{i:2d}] = {wi:+.6f}")
print(f"\nOptimal logistic loss  L* = {L_star:.6f}")
print(f"Per-sample loss        L*/n = {L_star / len(y):.6f}")
print(f"Training accuracy      = {model.score(X, y) * 100:.2f}%")
print(f"{'='*50}")

# ──────────────────────────────────────────────
# 5. Save L* and w* for downstream steps
# ──────────────────────────────────────────────
np.savetxt(os.path.join(BASE_DIR, "w_star.tsv"), w_star, delimiter="\t",
           header="w_star", comments="", fmt="%.10f")
with open(os.path.join(BASE_DIR, "L_star.txt"), "w") as f:
    f.write(f"{L_star:.10f}\n")

print(f"\nSaved w_star.tsv  shape={w_star.shape}")
print(f"Saved L_star.txt  (L* = {L_star:.6f})")

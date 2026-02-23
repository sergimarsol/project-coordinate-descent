# Project 2 — Coordinate Descent for Logistic Regression

Binary logistic regression on the UCI Wine dataset (classes 1 vs 2),
solved via coordinate descent with Gauss–Southwell selection rules.

## Requirements

- Python 3
- NumPy
- scikit-learn
- matplotlib

## Data

The `wine/` folder contains the original UCI Wine dataset (`wine.data`).
No manual download is needed.

## How to Run

Run the four scripts **in order**. Each depends on the output of the previous one.

```bash
python3 data_preprocessing.py
python3 baseline_logistic.py
python3 coordinate_descent.py
python3 sparse_coordinate_descent.py
```

## Script Descriptions

### 1. data_preprocessing.py
Loads `wine/wine.data`, keeps only classes 1 and 2, converts labels
(1 → +1, 2 → −1), and standardizes all 13 features to zero mean
and unit variance.

**Outputs:**
- `X.tsv` — Standardized feature matrix (130×13) with header row.
- `y.tsv` — Label column (+1/−1) with header row.

### 2. baseline_logistic.py
Trains sklearn LogisticRegression with C=1e10 (effectively unregularized)
to compute the optimal logistic loss L*.

**Outputs:**
- `w_star.tsv` — Optimal weight vector (13 values).
- `L_star.txt` — Scalar optimal loss value.

### 3. coordinate_descent.py
Runs 5000 iterations of three coordinate descent methods:
- **Gauss–Southwell (GS):** selects coordinate with largest |∂L/∂w_i|.
- **Normalized GS:** selects by |∂L/∂w_i| / √L_i (accounts for curvature).
- **Random:** selects coordinate uniformly at random.

All use the same update rule: w_i -= (1/L_i) * ∂L/∂w_i.

**Outputs:**
- `cd_losses.tsv` — Loss at each iteration for all three methods.
- `convergence_plot.png` — Log-scale convergence curves with L* baseline.

### 4. sparse_coordinate_descent.py
Runs k-sparse coordinate descent with an active set constraint (|S| ≤ k)
using the normalized GS rule. Tests k = 1, 3, 5, 7, 10, 13.

**Outputs:**
- `sparse_results.tsv` — Final loss, gap to L*, and active features per k.
- `sparse_losses.tsv` — Loss at each iteration for each k value.
- `sparse_convergence_plot.png` — Log-scale convergence curves per k.
- `sparse_loss_bar.png` — Bar chart of final loss vs sparsity level.

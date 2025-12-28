# ============================================================
# KNN FROM SCRATCH (NUMPY CORE) + UCI MAMMOGRAPHIC MASS DEMO
# ============================================================
# Core model: NumPy only.
# Demo/visualization: matplotlib.
# TF interoperability: optional wrapper using tf.numpy_function.
#
# Dataset (UCI Mammographic Mass):
# Features: BI-RADS, Age, Shape, Margin, Density
# Target: Severity (0=benign, 1=malignant)
# Missing values are present in several attributes.  :contentReference[oaicite:1]{index=1}
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)

def train_test_split_np(X, y, test_size=0.2, seed=42, stratify=True):
    """
    NumPy train/test split.
    If stratify=True (recommended for classification), preserves class proportions.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)

    if stratify:
        idx0 = idx[y == 0]
        idx1 = idx[y == 1]
        rng.shuffle(idx0); rng.shuffle(idx1)

        n_test0 = int(round(len(idx0) * test_size))
        n_test1 = int(round(len(idx1) * test_size))

        test_idx = np.concatenate([idx0[:n_test0], idx1[:n_test1]])
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)

        rng.shuffle(train_idx); rng.shuffle(test_idx)
    else:
        rng.shuffle(idx)
        n_test = int(round(n * test_size))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def nanmedian_impute_fit(X):
    """Compute per-feature median ignoring NaNs."""
    return np.nanmedian(X, axis=0)

def nanmedian_impute_transform(X, med):
    """Replace NaNs by the fitted median per column."""
    X2 = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X2[:, j])
        X2[m, j] = med[j]
    return X2

def standardize_fit(X):
    """Compute mean/std per feature."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return mu, sigma

def standardize_transform(X, mu, sigma):
    """(X - mean) / std."""
    return (X - mu) / sigma

# ----------------------------
# NumPy-only KNN implementation
# ----------------------------
class KNN:
    """
    NumPy-only KNN for classification/regression, with extras:

    Extras (beyond typical sklearn KNN "feel"):
    1) missing-aware Minkowski distance (missing='ignore'):
       - ignores dimensions where either sample is NaN
       - samples with no overlap get distance = +inf (never selected)
    2) nominal feature handling:
       - for specified columns, distance contribution is 0 if equal, 1 if different
    3) diagonal metric learning (very fast, stable):
       - classification: uses effect-size separation between class means per feature
       - regression: uses |corr(feature, target)|
       - yields per-feature weights for a diagonal Mahalanobis-like metric
    4) return_neighbors=True:
       - returns (neighbor_indices, neighbor_distances) for debugging/explainability
    5) explain=True:
       - returns a simple local feature “importance”: avg |x_neighbor - x_query| per feature
    6) conformal_set (practical set-valued prediction):
       - returns label sets by probability thresholding (useful uncertainty primitive)
    """

    def __init__(
        self,
        k=5,
        task="classification",       # "classification" or "regression"
        p=2.0,                       # Minkowski p (2=Euclidean, 1=Manhattan)
        metric="minkowski",          # "minkowski" or "cosine"
        weights="uniform",           # "uniform", "distance", "gaussian"
        distance_power=1.0,          # for distance weights: w = 1/(d+eps)^power
        gaussian_sigma=1.0,          # for gaussian weights
        eps=1e-12,
        nominal_idx=None,            # columns treated as nominal mismatch
        missing="ignore",            # "ignore" or "impute"
        learn_diagonal_metric=False,
        metric_strength=1.0          # >1 emphasizes informative dims more strongly
    ):
        self.k = int(k)
        self.task = task
        self.p = float(p)
        self.metric = metric
        self.weights = weights
        self.distance_power = float(distance_power)
        self.gaussian_sigma = float(gaussian_sigma)
        self.eps = float(eps)
        self.nominal_idx = set([] if nominal_idx is None else list(nominal_idx))
        self.missing = missing
        self.learn_diagonal_metric = bool(learn_diagonal_metric)
        self.metric_strength = float(metric_strength)

        self.X_ = None
        self.y_ = None
        self.diag_W_ = None          # learned diagonal weights (shape [d])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")
        if self.k <= 0 or self.k > X.shape[0]:
            raise ValueError("k must be in [1, n_train]")

        self.X_ = X
        self.y_ = y

        # Learn per-feature weights (optional). If disabled, weights are all ones.
        self.diag_W_ = self._learn_diagonal_weights(X, y) if self.learn_diagonal_metric else np.ones(X.shape[1])
        return self

    def _learn_diagonal_weights(self, X, y):
        """
        Very simple diagonal metric learning:
        - classification: effect size (Cohen's d-like) between class means / pooled std
        - regression: absolute correlation proxy
        """
        d = X.shape[1]
        eps = self.eps

        if self.task == "classification":
            classes = np.unique(y)
            if classes.size == 2:
                X0 = X[y == classes[0]]
                X1 = X[y == classes[1]]
                mu0 = np.nanmean(X0, axis=0); mu1 = np.nanmean(X1, axis=0)
                s0 = np.nanstd(X0, axis=0);  s1 = np.nanstd(X1, axis=0)
                pooled = np.sqrt((s0**2 + s1**2) / 2.0) + eps
                effect = np.abs(mu1 - mu0) / pooled
                w = (effect + eps) ** self.metric_strength
            else:
                # multi-class fallback: average one-vs-rest learned weights
                scores = np.zeros(d, dtype=float)
                for c in classes:
                    scores += self._learn_diagonal_weights(X, (y == c).astype(int))
                w = scores / max(classes.size, 1)
        else:
            # regression
            y2 = y.astype(float)
            y2 = (y2 - np.mean(y2)) / (np.std(y2) + eps)
            scores = np.zeros(d, dtype=float)

            for j in range(d):
                xj = X[:, j]
                m = ~np.isnan(xj)
                if np.sum(m) < 3:
                    scores[j] = 0.0
                    continue
                xj2 = (xj[m] - np.mean(xj[m])) / (np.std(xj[m]) + eps)
                scores[j] = np.abs(np.mean(xj2 * y2[m]))  # correlation proxy

            w = (scores + eps) ** self.metric_strength

        # Normalize so average weight ~1 (keeps distance scales stable).
        w = w / (np.mean(w) + eps)
        return w

    def _pairwise_distances(self, X_query):
        """
        Compute distance(query_i, train_j) for all i,j.

        Shape:
          X_query: [n_q, d] or [d]
          return D: [n_q, n_train]
        """
        Xq = np.asarray(X_query, dtype=float)
        Xt = self.X_
        if Xq.ndim == 1:
            Xq = Xq[None, :]

        n_q, d = Xq.shape
        n_t = Xt.shape[0]
        if d != Xt.shape[1]:
            raise ValueError("dimension mismatch")

        # Apply diagonal metric: scale each feature by sqrt(weight).
        sqrtW = np.sqrt(self.diag_W_)[None, :]
        XqW = Xq * sqrtW
        XtW = Xt * sqrtW

        if self.metric == "cosine":
            # Cosine is easiest if we already imputed missing values.
            if self.missing == "ignore":
                raise ValueError("cosine + missing='ignore' not supported; impute first")
            Xq2 = np.nan_to_num(XqW, nan=0.0)
            Xt2 = np.nan_to_num(XtW, nan=0.0)
            qn = np.linalg.norm(Xq2, axis=1, keepdims=True) + self.eps
            tn = np.linalg.norm(Xt2, axis=1, keepdims=True) + self.eps
            sim = (Xq2 @ Xt2.T) / (qn @ tn.T)
            return 1.0 - sim

        # Minkowski distances
        D = np.empty((n_q, n_t), dtype=float)

        for i in range(n_q):
            qi = XqW[i]
            diff = XtW - qi[None, :]  # broadcast to [n_t, d]

            if self.missing == "ignore":
                # mask dims that are valid (not NaN) for BOTH points
                mask = ~np.isnan(diff)

                # nominal columns: mismatch => contribution 1 (scaled)
                if self.nominal_idx:
                    for j in self.nominal_idx:
                        mj = mask[:, j]
                        if np.any(mj):
                            a = self.X_[:, j][mj]
                            b = Xq[i, j]
                            diff[mj, j] = (a != b).astype(float) * sqrtW[0, j]

                # compute Minkowski only over present dims
                present = np.sum(mask, axis=1)
                ad = np.abs(np.where(mask, diff, 0.0))
                dist_p = np.sum(ad ** self.p, axis=1)
                dist = dist_p ** (1.0 / self.p)

                # if no dims present, distance is inf (cannot be neighbor)
                D[i] = np.where(present > 0, dist, np.inf)

            else:
                # missing == "impute": treat NaNs as 0 (assumes upstream imputation/standardization)
                diff2 = np.nan_to_num(diff, nan=0.0)

                if self.nominal_idx:
                    for j in self.nominal_idx:
                        a = self.X_[:, j]
                        b = Xq[i, j]
                        diff2[:, j] = (a != b).astype(float) * sqrtW[0, j]

                ad = np.abs(diff2)
                D[i] = (np.sum(ad ** self.p, axis=1) + self.eps) ** (1.0 / self.p)

        return D

    def _neighbor_indices_and_distances(self, X_query):
        """
        Fast top-k neighbor selection:
          - argpartition selects k smallest in O(n)
          - then sort those k neighbors
        """
        D = self._pairwise_distances(X_query)
        k = self.k

        idx = np.argpartition(D, kth=k-1, axis=1)[:, :k]  # [n_q, k]
        row = np.arange(D.shape[0])[:, None]
        dist_k = D[row, idx]
        order = np.argsort(dist_k, axis=1)

        idx_sorted = idx[row, order]
        dist_sorted = dist_k[row, order]
        return idx_sorted, dist_sorted

    def _weights_from_distances(self, dist):
        if self.weights == "uniform":
            return np.ones_like(dist)
        if self.weights == "distance":
            return 1.0 / ((dist + self.eps) ** self.distance_power)
        if self.weights == "gaussian":
            s2 = (self.gaussian_sigma ** 2) + self.eps
            return np.exp(-(dist ** 2) / (2.0 * s2))
        raise ValueError("unknown weights")

    def predict(self, X_query, return_neighbors=False, return_proba=False, explain=False):
        """
        Returns:
          - regression: yhat
          - classification: yhat (and optionally proba)

        Optional:
          - return_neighbors: adds (idx, dist)
          - explain: adds feat_imp (avg |neighbor - query| per feature)
        """
        idx, dist = self._neighbor_indices_and_distances(X_query)
        neigh_y = self.y_[idx]
        w = self._weights_from_distances(dist)

        outputs = []

        if self.task == "regression":
            # weighted average
            yhat = np.sum(w * neigh_y, axis=1) / (np.sum(w, axis=1) + self.eps)
            outputs.append(yhat)
        else:
            # weighted vote
            classes = np.unique(self.y_)
            votes = np.zeros((idx.shape[0], classes.size), dtype=float)

            for ci, c in enumerate(classes):
                votes[:, ci] = np.sum(w * (neigh_y == c), axis=1)

            proba = votes / (np.sum(votes, axis=1, keepdims=True) + self.eps)
            yhat = classes[np.argmax(proba, axis=1)]

            outputs.append(yhat)
            if return_proba:
                outputs.append(proba)

        if explain:
            Xq = np.asarray(X_query, dtype=float)
            if Xq.ndim == 1:
                Xq = Xq[None, :]
            Xn = self.X_[idx]  # [n_q, k, d]
            diff = np.abs(Xn - Xq[:, None, :])
            diff = np.where(np.isnan(diff), np.nan, diff)
            feat_imp = np.nanmean(diff, axis=1)  # [n_q, d]
            outputs.append(feat_imp)

        if return_neighbors:
            outputs.append(idx)
            outputs.append(dist)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def conformal_set(self, X_query, alpha=0.1):
        """
        Practical set-valued prediction:
        return labels whose probability >= alpha.
        (Full conformal needs a calibration split; this is a useful lightweight alternative.)
        """
        yhat, proba = self.predict(X_query, return_proba=True)
        classes = np.unique(self.y_)
        sets = []
        for i in range(proba.shape[0]):
            sets.append(classes[proba[i] >= alpha])
        return sets

# ----------------------------
# Metrics (NumPy)
# ----------------------------
def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    m = np.zeros((labels.size, labels.size), dtype=int)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            m[i, j] = int(np.sum((y_true == a) & (y_pred == b)))
    return labels, m

def precision_recall_f1(y_true, y_pred, positive=1):
    tp = np.sum((y_true == positive) & (y_pred == positive))
    fp = np.sum((y_true != positive) & (y_pred == positive))
    fn = np.sum((y_true == positive) & (y_pred != positive))
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    return float(prec), float(rec), float(f1)

def plot_confusion_matrix(cm, labels, title="Confusion matrix"):
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.show()

# ----------------------------
# UCI Mammographic Mass: download + parse
# ----------------------------
def parse_mammographic_mass(lines):
    """
    Each line: bi-rads, age, shape, margin, density, severity
    Missing values are '?', converted to np.nan.
    """
    X_list = []
    y_list = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        *feat, target = parts
        row = [np.nan if v == "?" else float(v) for v in feat]
        if target == "?":
            continue
        X_list.append(row)
        y_list.append(int(float(target)))
    return np.array(X_list, dtype=float), np.array(y_list, dtype=int)

def main():
    set_seed(42)

    # Download the raw data file from UCI's classic hosting path.
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
    local_path = Path("mammographic_masses.data")
    if not local_path.exists():
        urllib.request.urlretrieve(data_url, local_path.as_posix())

    raw_lines = local_path.read_text().strip().splitlines()
    X_raw, y = parse_mammographic_mass(raw_lines)

    # Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split_np(
        X_raw, y, test_size=0.2, seed=42, stratify=True
    )

    # Impute (median) + standardize
    med = nanmedian_impute_fit(X_train_raw)
    X_train_imp = nanmedian_impute_transform(X_train_raw, med)
    X_test_imp  = nanmedian_impute_transform(X_test_raw,  med)

    mu, sigma = standardize_fit(X_train_imp)
    X_train = standardize_transform(X_train_imp, mu, sigma)
    X_test  = standardize_transform(X_test_imp,  mu, sigma)

    # Train KNN:
    # Treat Shape (col 2) and Margin (col 3) as nominal mismatch (0/1 penalty).
    knn = KNN(
        k=11,
        task="classification",
        weights="distance",
        distance_power=2.0,
        nominal_idx=[2, 3],
        missing="impute",              # we already imputed
        learn_diagonal_metric=True     # diagonal metric learning enabled
    ).fit(X_train, y_train)

    # Predict with extras: probabilities, explanations, neighbors
    y_pred, proba, feat_imp, neigh_idx, neigh_dist = knn.predict(
        X_test, return_proba=True, explain=True, return_neighbors=True
    )

    # Metrics
    acc = accuracy(y_test, y_pred)
    prec, rec, f1 = precision_recall_f1(y_test, y_pred, positive=1)

    print("Accuracy:", acc)
    print("Precision:", prec, "Recall:", rec, "F1:", f1)

    labels, cm = confusion_matrix(y_test, y_pred, labels=np.array([0, 1]))
    plot_confusion_matrix(cm, labels, title="Mammographic Mass: KNN confusion matrix")

    # Example: set-valued predictions (uncertainty)
    sets = knn.conformal_set(X_test[:10], alpha=0.15)
    print("Set-valued predictions for first 10 test samples:", sets)

    # ----------------------------
    # Complex visualization: PCA -> 2D decision regions (NumPy PCA)
    # ----------------------------
    def pca_fit(X, n_components=2):
        mu = np.mean(X, axis=0, keepdims=True)
        Xc = X - mu
        C = (Xc.T @ Xc) / (Xc.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1]
        W = eigvecs[:, order[:n_components]]
        return mu, W

    def pca_transform(X, mu, W):
        return (X - mu) @ W

    mu_pca, W_pca = pca_fit(X_train, n_components=2)
    Z_train = pca_transform(X_train, mu_pca, W_pca)
    Z_test  = pca_transform(X_test,  mu_pca, W_pca)

    knn2d = KNN(k=21, task="classification", weights="distance", distance_power=2.0, missing="impute").fit(Z_train, y_train)

    pad = 0.5
    x_min, x_max = Z_train[:, 0].min() - pad, Z_train[:, 0].max() + pad
    y_min, y_max = Z_train[:, 1].min() - pad, Z_train[:, 1].max() + pad

    grid_n = 250
    xs = np.linspace(x_min, x_max, grid_n)
    ys = np.linspace(y_min, y_max, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]

    grid_pred = knn2d.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, grid_pred, alpha=0.35)
    plt.scatter(Z_test[:, 0], Z_test[:, 1], c=y_test, s=25)
    plt.title("Decision regions (PCA 2D) + test points")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # TensorFlow interoperability (optional)
    # ----------------------------
    try:
        import tensorflow as tf

        @tf.function
        def knn_predict_tf(x_batch):
            # Calls NumPy code inside TF graph
            y_out = tf.numpy_function(func=lambda a: knn.predict(a), inp=[x_batch], Tout=tf.int64)
            y_out.set_shape([None])
            return y_out

        print("TF wrapper predictions:", knn_predict_tf(tf.constant(X_test[:8], dtype=tf.float32)).numpy())
    except Exception as e:
        print("TensorFlow not available here, but tf.numpy_function wrapper pattern is correct.")
        print("Error:", e)

if __name__ == "__main__":
    main()

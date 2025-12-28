import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path
from copy import deepcopy

# ============================================================
# Utilities
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)

def train_test_split_np(X, y, test_size=0.2, seed=42, stratify=True):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))

    if stratify:
        idx0, idx1 = idx[y == 0], idx[y == 1]
        rng.shuffle(idx0); rng.shuffle(idx1)
        n0, n1 = int(len(idx0)*test_size), int(len(idx1)*test_size)
        test_idx = np.concatenate([idx0[:n0], idx1[:n1]])
        train_idx = np.setdiff1d(idx, test_idx)
    else:
        rng.shuffle(idx)
        n = int(len(idx)*test_size)
        test_idx, train_idx = idx[:n], idx[n:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def nanmedian_impute(X):
    med = np.nanmedian(X, axis=0)
    X2 = X.copy()
    for j in range(X.shape[1]):
        X2[np.isnan(X2[:, j]), j] = med[j]
    return X2

def standardize(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    return (X - mu) / sd

# ============================================================
# Base KNN (NumPy-only)
# ============================================================

class KNN:
    def __init__(self, k=11, weights="distance", eps=1e-12):
        self.k = k
        self.weights = weights
        self.eps = eps

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _dist(self, x):
        return np.linalg.norm(self.X - x, axis=1)

    def predict_proba(self, Xq):
        probs = []
        for x in Xq:
            d = self._dist(x)
            idx = np.argsort(d)[:self.k]
            w = 1.0 / (d[idx] + self.eps)
            classes = np.unique(self.y)
            p = np.zeros(len(classes))
            for i,c in enumerate(classes):
                p[i] = np.sum(w * (self.y[idx] == c))
            probs.append(p / p.sum())
        return classes, np.array(probs)

    def predict(self, Xq):
        cls, p = self.predict_proba(Xq)
        return cls[np.argmax(p, axis=1)]

# ============================================================
# Multi-probe Random Hyperplane LSH
# ============================================================

def hamming_neighbors(code, bits, r):
    out = {code}
    for i in range(bits):
        out.add(code ^ (1 << i))
    if r >= 2:
        for i in range(bits):
            for j in range(i+1, bits):
                out.add(code ^ (1<<i) ^ (1<<j))
    return out

class RandomHyperplaneLSH:
    def __init__(self, tables=10, bits=14, radius=2):
        self.tables = tables
        self.bits = bits
        self.radius = radius

    def fit(self, X):
        self.X = X
        d = X.shape[1]
        self.R = np.random.randn(self.tables, self.bits, d)
        self.hash_tables = []

        for t in range(self.tables):
            codes = self._hash(X, self.R[t])
            table = {}
            for i,c in enumerate(codes):
                table.setdefault(int(c), []).append(i)
            self.hash_tables.append(table)
        return self

    def _hash(self, X, R):
        proj = X @ R.T
        bits = (proj >= 0).astype(np.uint32)
        codes = np.zeros(X.shape[0], dtype=np.uint32)
        for b in range(bits.shape[1]):
            codes |= bits[:, b] << b
        return codes

    def query(self, x):
        x = x.reshape(1,-1)
        cand = set()
        for t in range(self.tables):
            code = int(self._hash(x, self.R[t])[0])
            for c in hamming_neighbors(code, self.bits, self.radius):
                cand.update(self.hash_tables[t].get(c, []))
        return np.array(list(cand))

# ============================================================
# Approximate KNN using Multi-probe LSH
# ============================================================

class ApproximateKNN:
    def __init__(self, knn, lsh, min_candidates=80):
        self.knn = knn
        self.lsh = lsh
        self.min_candidates = min_candidates

    def fit(self, X, y):
        self.X, self.y = X, y
        self.lsh.fit(X)
        self.knn.fit(X, y)
        return self

    def predict(self, Xq):
        preds = []
        for x in Xq:
            cand = self.lsh.query(x)
            if len(cand) < self.min_candidates:
                extra = np.random.choice(len(self.X), self.min_candidates, replace=False)
                cand = np.unique(np.concatenate([cand, extra]))
            sub = KNN(self.knn.k).fit(self.X[cand], self.y[cand])
            preds.append(sub.predict(x.reshape(1,-1))[0])
        return np.array(preds)

# ============================================================
# Split Conformal Prediction
# ============================================================

class ConformalKNN:
    def __init__(self, knn, alpha=0.1):
        self.knn = knn
        self.alpha = alpha

    def fit(self, Xtr, ytr, Xcal, ycal):
        self.knn.fit(Xtr, ytr)
        cls, p = self.knn.predict_proba(Xcal)
        idx = {c:i for i,c in enumerate(cls)}
        scores = 1 - np.array([p[i, idx[ycal[i]]] for i in range(len(ycal))])
        scores.sort()
        k = int(np.ceil((len(scores)+1)*(1-self.alpha))) - 1
        self.qhat = scores[np.clip(k,0,len(scores)-1)]
        self.classes = cls
        return self

    def predict_set(self, Xq):
        cls, p = self.knn.predict_proba(Xq)
        return [cls[(1-p[i]) <= self.qhat] for i in range(len(Xq))]

# ============================================================
# Diagnostics
# ============================================================

def conformal_diagnostics(Xtr, ytr, Xte, yte):
    alphas = np.linspace(0.05,0.3,10)
    cover, size = [], []

    idx = np.random.permutation(len(Xtr))
    Xp, Xc = Xtr[idx[:len(idx)//2]], Xtr[idx[len(idx)//2:]]
    yp, yc = ytr[idx[:len(idx)//2]], ytr[idx[len(idx)//2:]]

    base = KNN(11)
    for a in alphas:
        conf = ConformalKNN(deepcopy(base), a).fit(Xp, yp, Xc, yc)
        sets = conf.predict_set(Xte)
        cover.append(np.mean([yte[i] in sets[i] for i in range(len(yte))]))
        size.append(np.mean([len(s) for s in sets]))

    plt.plot(alphas, cover, label="empirical")
    plt.plot(alphas, 1-alphas, "--", label="target")
    plt.legend(); plt.title("Coverage vs alpha"); plt.show()

    plt.plot(alphas, size); plt.title("Mean set size"); plt.show()

# ============================================================
# Main Experiment (Mammographic Mass)
# ============================================================

def load_mammographic():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
    path = Path("mammographic.data")
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    X, y = [], []
    for l in path.read_text().splitlines():
        p = l.split(",")
        if "?" in p: continue
        X.append(list(map(float, p[:-1])))
        y.append(int(p[-1]))
    return np.array(X), np.array(y)

if __name__ == "__main__":
    set_seed()

    X, y = load_mammographic()
    X = standardize(nanmedian_impute(X))

    Xtr, Xte, ytr, yte = train_test_split_np(X, y)

    knn = KNN(11)
    lsh = RandomHyperplaneLSH()
    approx = ApproximateKNN(knn, lsh).fit(Xtr, ytr)

    print("Approx-KNN accuracy:", np.mean(approx.predict(Xte) == yte))

    conformal_diagnostics(Xtr, ytr, Xte, yte)

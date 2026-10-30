"""Anomaly detection pipeline: Classical ML + Deep Learning + Quantum ML.

Models (all trained once on the labelled demo dataset, then applied to uploads):
  * Linear SVM  - classical ML  (scikit-learn LinearSVC)
  * LSTM        - deep learning (NumPy LSTM with BPTT; features fed as a sequence)
  * VQC         - quantum ML    (4-qubit variational circuit, simulated in NumPy)
  * Isolation Forest - unsupervised baseline
Final label = majority vote of the three supervised models.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

LABELS = {"label", "target", "class", "anomaly", "is_anomaly"}
FEATURES = ["event_count", "sequence_length", "duration", "error_rate"]


def demo_dataframe(seed=7, n=80, na=8):
    r = np.random.default_rng(seed)
    X = np.vstack([r.normal(0, 1, (n, 4)), r.normal(4.5, .7, (na, 4))])
    d = pd.DataFrame(X, columns=FEATURES)
    d["label"] = ["Normal"] * n + ["Anomaly"] * na
    return d


# ---------------------------------------------------------------- Linear SVM
class SVMModel:
    name = "Linear SVM"

    def fit(self, X, y):
        self.m = make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", random_state=42, max_iter=10000))
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def score(self, X):
        return self.m.decision_function(X)


# ---------------------------------------------------------------------- LSTM
class LSTMModel:
    """Single-layer LSTM (NumPy, manual backprop-through-time + Adam).
    The 4 standardised features are fed as a length-4 sequence."""
    name = "LSTM"
    H = 8

    @staticmethod
    def _sig(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    def _forward(self, X):
        B, T = X.shape
        H = self.H
        h = np.zeros((B, H)); c = np.zeros((B, H)); cache = []
        for t in range(T):
            z = np.concatenate([X[:, t:t + 1], h], axis=1)
            g = z @ self.W + self.b
            i, f, o = self._sig(g[:, :H]), self._sig(g[:, H:2 * H]), self._sig(g[:, 2 * H:3 * H])
            u = np.tanh(g[:, 3 * H:])
            c_new = f * c + i * u
            h_new = o * np.tanh(c_new)
            cache.append((z, i, f, o, u, c, c_new))
            h, c = h_new, c_new
        logit = h @ self.V + self.d
        return logit, h, cache

    def fit(self, X, y, epochs=300, lr=0.02):
        self.sc = StandardScaler().fit(X)
        Xs = self.sc.transform(X); H = self.H
        r = np.random.default_rng(42)
        self.W = r.normal(0, .3, (1 + H, 4 * H)); self.b = np.zeros(4 * H); self.b[H:2 * H] = 1.0
        self.V = r.normal(0, .3, H); self.d = 0.0
        params = ["W", "b", "V", "d"]
        m = {k: 0 for k in params}; v = {k: 0 for k in params}
        sw = np.where(y == 1, (y == 0).sum() / max(1, (y == 1).sum()), 1.0)
        for ep in range(1, epochs + 1):
            logit, h, cache = self._forward(Xs)
            p = self._sig(logit)
            dl = (sw * (p - y)) / len(y)
            gV = (h * dl[:, None]).sum(0); gd = dl.sum()
            dh = dl[:, None] * self.V[None, :]; dc = np.zeros_like(dh)
            gW = np.zeros_like(self.W); gb = np.zeros_like(self.b)
            for t in reversed(range(len(cache))):
                z, i, f, o, u, c_prev, c_new = cache[t]
                tc = np.tanh(c_new)
                do = dh * tc
                dc = dc + dh * o * (1 - tc ** 2)
                di, df, du = dc * u, dc * c_prev, dc * i
                dg = np.concatenate([di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), du * (1 - u ** 2)], axis=1)
                gW += z.T @ dg; gb += dg.sum(0)
                dz = dg @ self.W.T
                dh = dz[:, 1:]; dc = dc * f
            grads = {"W": gW, "b": gb, "V": gV, "d": gd}
            for k in params:
                m[k] = .9 * m[k] + .1 * grads[k]; v[k] = .999 * v[k] + .001 * grads[k] ** 2
                mh = m[k] / (1 - .9 ** ep); vh = v[k] / (1 - .999 ** ep)
                setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + 1e-8))
        return self

    def score(self, X):
        logit, _, _ = self._forward(self.sc.transform(X))
        return self._sig(logit)

    def predict(self, X):
        return (self.score(X) > 0.5).astype(int)


# ----------------------------------------------------------------------- VQC
class VQCModel:
    """Variational quantum classifier: 4 qubits, angle encoding, 2 layers of
    RY rotations + CNOT ring, output = <Z> on qubit 0. Simulated with NumPy
    state vectors; trained with the parameter-shift rule."""
    name = "VQC (quantum)"
    NQ, LAYERS = 4, 2

    @staticmethod
    def _ry(t):
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -s], [s, c]])

    def _apply1(self, st, U, q):
        st = np.moveaxis(st, q + 1, 1)
        st = np.tensordot(st, U, axes=([1], [1]))  # (B, rest..., 2)
        st = np.moveaxis(st, -1, 1)
        return np.moveaxis(st, 1, q + 1)

    def _cnot(self, st, c, t):
        idx = [slice(None)] * st.ndim
        idx[c + 1] = 1
        sub = st[tuple(idx)]
        ax = t + 1 - (1 if t > c else 0)
        st = st.copy()
        st[tuple(idx)] = np.flip(sub, axis=ax)
        return st

    def _expval(self, X, w):
        B = len(X)
        st = np.zeros((B,) + (2,) * self.NQ)
        st[(slice(None),) + (0,) * self.NQ] = 1.0
        for q in range(self.NQ):  # encoding: per-sample RY(x_q)
            c, s = np.cos(X[:, q] / 2), np.sin(X[:, q] / 2)
            U = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (B,2,2)
            st = np.moveaxis(st, q + 1, -1)
            st = np.einsum("b...j,bij->b...i", st, U)
            st = np.moveaxis(st, -1, q + 1)
        w = w.reshape(self.LAYERS, self.NQ)
        for l in range(self.LAYERS):
            for q in range(self.NQ):
                st = self._apply1(st, self._ry(w[l, q]), q)
            for q in range(self.NQ):
                st = self._cnot(st, q, (q + 1) % self.NQ)
        p = (st ** 2)
        p0 = p.sum(axis=tuple(range(2, self.NQ + 1)))  # prob of qubit0 in |0>,|1>
        return p0[:, 0] - p0[:, 1]  # <Z> in [-1, 1]

    def fit(self, X, y, iters=60, lr=0.4):
        self.sc = StandardScaler().fit(X)
        Z = np.tanh(self.sc.transform(X)) * np.pi / 2
        t = np.where(y == 1, -1.0, 1.0)  # anomaly -> -1
        rng = np.random.default_rng(42)
        w = rng.normal(0, 0.5, self.LAYERS * self.NQ)
        cw = np.where(y == 1, (y == 0).sum() / max(1, (y == 1).sum()), 1.0)
        for _ in range(iters):
            f = self._expval(Z, w)
            g = np.zeros_like(w)
            for i in range(len(w)):
                wp, wm = w.copy(), w.copy()
                wp[i] += np.pi / 2
                wm[i] -= np.pi / 2
                df = (self._expval(Z, wp) - self._expval(Z, wm)) / 2
                g[i] = np.mean(cw * 2 * (f - t) * df)
            w -= lr * g
        self.w = w
        return self

    def score(self, X):
        Z = np.tanh(self.sc.transform(X)) * np.pi / 2
        return -self._expval(Z, self.w)  # higher = more anomalous

    def predict(self, X):
        return (self.score(X) > 0).astype(int)


# ------------------------------------------------------------------ pipeline
def _xy(df):
    X = df[FEATURES].to_numpy(dtype=float)
    y = (df["label"].astype(str).str.lower() == "anomaly").astype(int).to_numpy()
    return X, y


def _metrics(y, p):
    return {"accuracy": round(float(accuracy_score(y, p)), 3), "f1": round(float(f1_score(y, p, zero_division=0)), 3)}


@lru_cache(maxsize=1)
def trained_models():
    X, y = _xy(demo_dataframe(seed=7, n=400, na=100))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    models, metrics = [], {}
    for cls in (SVMModel, LSTMModel, VQCModel):
        m = cls().fit(Xtr, ytr)
        metrics[m.name] = _metrics(yte, m.predict(Xte))
        models.append(m)
    return models, metrics


TAGS = {"Linear SVM": "svm", "LSTM": "lstm", "VQC (quantum)": "vqc"}
KIND = {"Linear SVM": "Classical ML", "LSTM": "Deep Learning", "VQC (quantum)": "Quantum ML"}


def predict_dataframe(df):
    X = df.copy()
    X.columns = [str(c).strip().lower() for c in X.columns]
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    F = X[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    F = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(F), columns=FEATURES)
    Fv = F.to_numpy()

    lab = next((c for c in X.columns if c in LABELS), None)
    y_true = None
    if lab is not None:
        yy = (X[lab].astype(str).str.strip().str.lower().isin({"anomaly", "1", "true", "abnormal"})).astype(int).to_numpy()
        if yy.sum() >= 1 and (yy == 0).sum() >= 1 and len(yy) >= 4:
            y_true = yy

    if y_true is not None:   # train all three on the uploaded labelled data
        models = [cls().fit(Fv, y_true) for cls in (SVMModel, LSTMModel, VQCModel)]
        basis = "Trained on the uploaded labelled data; metrics are in-sample (vs. the label column)."
    else:                    # fall back to models trained on the synthetic demo set
        models, held = trained_models()
        basis = "Pre-trained on synthetic demo data (no usable label column); metrics are from a held-out split of that data."

    out = df.copy()
    votes = np.zeros(len(out), dtype=int)
    comparison = []
    for m in models:
        p = m.predict(Fv)
        votes += p
        out[f"{TAGS[m.name]}_prediction"] = np.where(p == 1, "Anomaly", "Normal")
        row = {"model": m.name, "type": KIND[m.name], "flagged": int(p.sum())}
        row.update(_metrics(y_true, p) if y_true is not None else held[m.name])
        comparison.append(row)

    iso = make_pipeline(StandardScaler(), IsolationForest(n_estimators=150, contamination="auto", random_state=42)).fit(Fv)
    ip = (iso.predict(Fv) == -1).astype(int)
    out["iforest_prediction"] = np.where(ip == 1, "Anomaly", "Normal")
    out["anomaly_score"] = (-iso.decision_function(Fv)).round(6)
    ens = (votes >= 2).astype(int)
    out["prediction"] = np.where(ens == 1, "Anomaly", "Normal")
    if y_true is not None:
        comparison.append({"model": "Isolation Forest", "type": "Baseline (unsupervised)", "flagged": int(ip.sum()), **_metrics(y_true, ip)})
        comparison.append({"model": "Ensemble (majority vote)", "type": "SVM + LSTM + VQC", "flagged": int(ens.sum()), **_metrics(y_true, ens)})
    else:
        comparison.append({"model": "Isolation Forest", "type": "Baseline (unsupervised)", "flagged": int(ip.sum()), "accuracy": "-", "f1": "-"})
        comparison.append({"model": "Ensemble (majority vote)", "type": "SVM + LSTM + VQC", "flagged": int(ens.sum()), "accuracy": "-", "f1": "-"})

    summary = {
        "rows": len(out),
        "anomalies": int(ens.sum()),
        "method": "Ensemble of Linear SVM (ML) + LSTM (DL) + VQC (QML), majority vote. " + basis,
        "comparison": comparison,
    }
    return out, summary

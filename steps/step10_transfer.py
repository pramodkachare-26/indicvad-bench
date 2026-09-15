"""Step 10 -- full RQ3: transfer geometry and adaptation economics.

Runs entirely on the cached features from Step 09, so the whole grid of
protocols x budgets x languages costs minutes rather than GPU-hours.

Protocols per held-out target language L
----------------------------------------
  matched   train on everything including L          (upper bound)
  lolo      train on the other 7 languages, 0 min L  (unseen-language penalty)
  lofo      train only on the other FAMILY           (typological penalty)
  lolo+n    lolo, then adapt the head on n minutes of L
  lofo+n    lofo, then adapt the head on n minutes of L

The paper claim this supports: how many minutes of target-language audio are
needed to close the unseen-language gap, and whether crossing a family
boundary costs more than merely being unseen.

Outputs
-------
  transfer_curves.csv     DetER per (target, protocol, budget)
  transfer_typology.csv   transfer gain vs typological distance

A WARNING YOU MUST CARRY INTO THE PAPER
---------------------------------------
The typology regression has n = 8 languages. Even using all 56 ordered pairs,
those pairs share training sets and are not independent. This is reported as a
correlation with an explicit power caveat, NOT as a headline result. No amount
of compute fixes it; it needs more languages.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from vadlib.config import lang_meta, resolve_device
from vadlib.decode import decode, resample_posterior
from vadlib.metrics import detection_metrics, roc_auc

SSL_FRAME_MS = 20
SCORE_FRAME_MS = 10


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def _load_split(index, langs=None, splits=None, max_minutes=None, seed=0,
                stride=1):
    """Concatenate cached features for a language/split selection."""
    sel = index
    if langs is not None:
        sel = sel[sel["lang"].isin(langs)]
    if splits is not None:
        sel = sel[sel["split"].isin(splits)]
    if sel.empty:
        return np.zeros((0, 1), np.float32), np.zeros(0, np.uint8), []

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sel))
    X, Y, used, minutes = [], [], [], 0.0
    for i in order:
        r = sel.iloc[int(i)]
        try:
            with np.load(r["path"]) as z:
                f = np.asarray(z["feat"], dtype=np.float32)[::stride]
                y = np.asarray(z["label"], dtype=np.uint8)[::stride]
        except Exception:
            continue
        n = min(len(f), len(y))
        X.append(f[:n]); Y.append(y[:n]); used.append(r["session_id"])
        minutes += n * stride * SSL_FRAME_MS / 1000.0 / 60.0
        if max_minutes is not None and minutes >= max_minutes:
            break
    if not X:
        return np.zeros((0, 1), np.float32), np.zeros(0, np.uint8), []
    return np.concatenate(X), np.concatenate(Y), used


class Head:
    """Small MLP over frozen SSL features. Trains in seconds on cached data.

    Uses torch when available (GPU-capable); falls back to sklearn's MLP
    otherwise, so the transfer experiment still runs on a machine without
    torch -- just slower. Both backends expose the same fit/predict contract,
    and `fit` can be called twice to implement adaptation: once on the source
    languages, then again on n minutes of target audio.
    """

    def __init__(self, dim, hidden=256, lr=1e-3, epochs=6, bs=1024,
                 device="cpu", seed=0):
        self.dim, self.hidden, self.lr = dim, int(hidden), float(lr)
        self.epochs, self.bs, self.seed = int(epochs), int(bs), int(seed)
        self.mu = self.sd = None
        self.backend = "torch"
        try:
            import torch
            import torch.nn as nn
            torch.manual_seed(seed)
            self.device = torch.device(device)
            self._torch, self._nn = torch, nn
            self.net = nn.Sequential(
                nn.Linear(dim, self.hidden), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(self.hidden, 1)).to(self.device)
        except Exception:
            from sklearn.neural_network import MLPClassifier
            self.backend = "sklearn"
            self.net = MLPClassifier(hidden_layer_sizes=(self.hidden,),
                                     learning_rate_init=self.lr,
                                     max_iter=self.epochs, warm_start=True,
                                     random_state=self.seed)

    def _norm(self, X, reset):
        if reset or self.mu is None:
            self.mu = X.mean(0, keepdims=True)
            self.sd = X.std(0, keepdims=True) + 1e-6
        return ((X - self.mu) / self.sd).astype(np.float32)

    def fit(self, X, Y, epochs=None, lr=None, reset_norm=True):
        if len(X) == 0:
            return self
        Xn = self._norm(X, reset_norm)
        if self.backend == "sklearn":
            self.net.set_params(max_iter=int(epochs or self.epochs),
                                learning_rate_init=float(lr or self.lr))
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.net.fit(Xn, Y.astype(int))
            return self
        torch, nn = self._torch, self._nn
        Xt = torch.from_numpy(Xn)
        Yt = torch.from_numpy(Y.astype(np.float32))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr or self.lr)
        lossf = nn.BCEWithLogitsLoss()
        self.net.train()
        for _ in range(int(epochs or self.epochs)):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), self.bs):
                b = perm[i:i + self.bs]
                xb = Xt[b].to(self.device)
                yb = Yt[b].to(self.device)
                opt.zero_grad()
                lossf(self.net(xb).squeeze(-1), yb).backward()
                opt.step()
        return self

    def predict(self, X):
        if len(X) == 0:
            return np.zeros(0, np.float32)
        Xn = ((X - self.mu) / self.sd).astype(np.float32)
        if self.backend == "sklearn":
            return self.net.predict_proba(Xn)[:, 1].astype(np.float32)
        torch = self._torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xn), 8192):
                xb = torch.from_numpy(Xn[i:i + 8192]).to(self.device)
                out.append(torch.sigmoid(self.net(xb).squeeze(-1)).cpu().numpy())
        return np.concatenate(out)


def _score(y_true, prob, cfg):
    """Score at the 10 ms grid so numbers are comparable with Step 05."""
    n10 = len(y_true) * SSL_FRAME_MS // SCORE_FRAME_MS
    ref = np.repeat(y_true.astype(bool), SSL_FRAME_MS // SCORE_FRAME_MS)[:n10]
    p10 = resample_posterior(prob, len(ref))
    hyp = decode(p10, SCORE_FRAME_MS, threshold=0.5)
    d = detection_metrics(ref, hyp, SCORE_FRAME_MS, 0.0,
                          float(cfg["scoring"]["dcf_miss_weight"]),
                          float(cfg["scoring"]["dcf_fa_weight"]))
    d["auc"] = roc_auc(ref, p10)
    return d


# --------------------------------------------------------------------------- #
# Typology
# --------------------------------------------------------------------------- #
def _typo_distance(cfg):
    """Gower distance over the typological attributes declared in config.

    Uses lang2vec/URIEL when installed; otherwise falls back to the family +
    laryngeal + gemination features already in config.yaml. The fallback is
    coarse and is labelled as such in the output.
    """
    meta = lang_meta(cfg)
    codes = [c for c, m in meta.items() if not m.get("control")]
    source = "config_attributes"
    try:
        import lang2vec.lang2vec as l2v
        iso = {c: c.split("_")[0] for c in codes}
        vecs = l2v.get_features(list(iso.values()), "syntax_knn+phonology_knn")
        M = np.array([vecs[iso[c]] for c in codes], dtype=float)
        D = np.zeros((len(codes), len(codes)))
        for i in range(len(codes)):
            for j in range(len(codes)):
                D[i, j] = np.mean(np.abs(M[i] - M[j]))
        source = "lang2vec_uriel"
        return codes, D, source
    except Exception:
        pass

    gem = {"low": 0.0, "mid": 0.5, "high": 1.0}
    feats = []
    for c in codes:
        m = meta[c]
        feats.append([1.0 if m.get("breathy_series") else 0.0,
                      gem.get(str(m.get("geminate_load", "mid")), 0.5)])
    F = np.array(feats)
    fam = [meta[c]["family"] for c in codes]
    D = np.zeros((len(codes), len(codes)))
    for i in range(len(codes)):
        for j in range(len(codes)):
            d = np.mean(np.abs(F[i] - F[j]))
            d += 1.0 if fam[i] != fam[j] else 0.0
            D[i, j] = d / 2.0
    return codes, D, source


# --------------------------------------------------------------------------- #
def run(cfg):
    ssl_cfg = cfg.get("ssl", {})
    if not ssl_cfg.get("enabled", False):
        print("[step10] ssl disabled -- skipping RQ3 transfer")
        return None
    idx_path = Path(cfg["paths"]["post"]) / "ssl" / "index.csv"
    if not idx_path.exists():
        print("[step10] no cached SSL features -- run step 09 first")
        return None
    device = resolve_device(cfg)
    index = pd.read_csv(idx_path)
    res = Path(cfg["paths"]["results"])
    tr_cfg = cfg["transfer"]
    budgets = list(tr_cfg["budgets_min"])
    stride = int(tr_cfg.get("frame_stride", 2))
    seed = int(cfg["project"]["seed"])

    langs = sorted(index["lang"].unique())
    fam = dict(zip(index["lang"], index["family"]))
    print(f"[step10] device={device} langs={len(langs)} budgets={budgets}")

    # Probe dimensionality once.
    with np.load(index.iloc[0]["path"]) as z:
        dim = int(np.asarray(z["feat"]).shape[1])
    probe = Head(dim, device=device)
    print(f"[step10] feature dim = {dim}  head backend = {probe.backend}")
    if probe.backend == "sklearn":
        print("[step10] torch not found; using sklearn MLP head (slower, CPU)")

    def new_head():
        return Head(dim, hidden=int(tr_cfg["hidden"]), lr=float(tr_cfg["lr"]),
                    epochs=int(tr_cfg["epochs"]), bs=int(tr_cfg["batch_size"]),
                    device=device, seed=seed)

    rows = []
    for L in langs:
        t0 = time.time()
        others = [x for x in langs if x != L]
        other_fam = [x for x in others if fam[x] != fam[L]]

        Xte, Yte, _ = _load_split(index, [L], ["test"], stride=stride)
        if len(Xte) == 0:
            continue

        base_sets = {"matched": langs, "lolo": others}
        if other_fam:
            base_sets["lofo"] = other_fam

        for proto, train_langs in base_sets.items():
            Xtr, Ytr, _ = _load_split(index, train_langs, ["dev", "test"],
                                      seed=seed, stride=stride)
            if len(Xtr) == 0:
                continue
            head = new_head().fit(Xtr, Ytr)
            d = _score(Yte, head.predict(Xte), cfg)
            rows.append({"target": L, "family": fam[L], "protocol": proto,
                         "budget_min": 0.0, "train_frames": len(Xtr), **d})
            print(f"  {L} {proto:8s} budget=0    DetER={d['deter']:.4f}")

            # Adaptation: reuse the pretrained head, fine-tune on n minutes of
            # target-language DEV audio only. Test sessions are never touched.
            if proto == "matched":
                continue
            for b in budgets:
                Xa, Ya, _ = _load_split(index, [L], ["dev"], max_minutes=b,
                                        seed=seed, stride=stride)
                if len(Xa) == 0:
                    continue
                ad = new_head().fit(Xtr, Ytr)
                ad.fit(Xa, Ya, epochs=int(tr_cfg["adapt_epochs"]),
                       lr=float(tr_cfg["adapt_lr"]), reset_norm=False)
                d2 = _score(Yte, ad.predict(Xte), cfg)
                rows.append({"target": L, "family": fam[L],
                             "protocol": f"{proto}+adapt", "budget_min": float(b),
                             "train_frames": len(Xtr), "adapt_frames": len(Xa),
                             **d2})
                print(f"  {L} {proto:8s} budget={b:<4g} DetER={d2['deter']:.4f}")
        print(f"  [{L}] {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[step10] no results produced")
        return None
    df.to_csv(res / "transfer_curves.csv", index=False)

    # ---- headline summary -------------------------------------------------
    piv = df[df.budget_min == 0].pivot_table(index="target", columns="protocol",
                                             values="deter")
    print("\n[step10] zero-shot DetER by protocol:")
    print(piv.round(4).to_string())
    if {"lolo", "matched"}.issubset(piv.columns):
        print(f"\n  mean unseen-language penalty : "
              f"{(piv['lolo'] - piv['matched']).mean():.4f}")
    if {"lofo", "lolo"}.issubset(piv.columns):
        print(f"  mean extra cost of crossing family: "
              f"{(piv['lofo'] - piv['lolo']).mean():.4f}")

    ad = df[df.protocol == "lolo+adapt"]
    if not ad.empty and "lolo" in piv.columns and "matched" in piv.columns:
        print("\n[step10] gap closed by n minutes of target audio:")
        for b, g in ad.groupby("budget_min"):
            m = g.set_index("target")["deter"]
            base = piv["lolo"].reindex(m.index)
            ceil = piv["matched"].reindex(m.index)
            denom = (base - ceil).replace(0, np.nan)
            frac = float(((base - m) / denom).mean())
            flag = "  (>100%: adapted head BEATS the matched multilingual " \
                   "head -- a specialist can outperform a generalist; " \
                   "report it, do not clamp it)" if frac > 1.0 else ""
            print(f"  {b:>5g} min : {frac:.1%}{flag}")

    # ---- typology (secondary, underpowered) -------------------------------
    codes, D, source = _typo_distance(cfg)
    tp = []
    zs = df[(df.budget_min == 0) & (df.protocol == "lolo")].set_index("target")
    mt = df[(df.budget_min == 0) & (df.protocol == "matched")].set_index("target")
    for L in codes:
        if L not in zs.index or L not in mt.index:
            continue
        i = codes.index(L)
        others = [j for j in range(len(codes)) if j != i]
        # NOTE: mean distance to all other languages is nearly CONSTANT in a
        # balanced design (every language has the same family split), so it has
        # no variance to correlate against. Distance to the NEAREST source
        # language is the quantity that actually varies and is the one that
        # should predict transfer difficulty.
        tp.append({"target": L, "family": fam.get(L),
                   "min_typological_distance": float(min(D[i, j] for j in others)),
                   "mean_typological_distance": float(np.mean([D[i, j] for j in others])),
                   "unseen_penalty": float(zs.loc[L, "deter"] - mt.loc[L, "deter"]),
                   "distance_source": source})
    tdf = pd.DataFrame(tp)
    if len(tdf) >= 3:
        x = tdf["min_typological_distance"].to_numpy(float)
        y = tdf["unseen_penalty"].to_numpy(float)
        if x.std() < 1e-12 or y.std() < 1e-12:
            r = float("nan")
            print(f"\n[step10] typology correlation undefined: "
                  f"{'distance' if x.std() < 1e-12 else 'penalty'} has zero "
                  "variance across languages. With a balanced family design "
                  "this is expected -- you need languages at varying "
                  "typological remove, not two tight clusters.")
        else:
            r = float(np.corrcoef(x, y)[0, 1])
            print(f"\n[step10] typology vs unseen penalty: r = {r:.3f} "
                  f"(n = {len(tdf)}, source = {source})")
        tdf["pearson_r"] = r
        tdf["n_languages"] = len(tdf)
        tdf["underpowered"] = len(tdf) < 15
        print("[step10] NOTE: n is far too small for an inferential claim. "
              "Report descriptively, with the caveat in the text.")
    tdf.to_csv(res / "transfer_typology.csv", index=False)
    print(f"[step10] wrote transfer_curves.csv, transfer_typology.csv")
    return res / "transfer_curves.csv"

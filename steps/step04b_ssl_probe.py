"""Step 04b (OPTIONAL, GPU) -- frozen SSL probe, trained leave-one-language-out.

WHY THIS IS A SEPARATE CATEGORY, NOT ANOTHER TABLE ROW
------------------------------------------------------
Every other system in this benchmark is ZERO-SHOT: it never sees the benchmark
before being scored on it. A probe trained on benchmark audio is not
comparable to those, and printing it as a peer row would flatter it unfairly.

So the probe is trained LEAVE-ONE-LANGUAGE-OUT: the head that scores Tamil is
trained only on the other twelve languages. It never sees a frame of Tamil.
That makes it an honest "supervised, in-domain, unseen-language" reference --
a *category*, reported separately in the systems table, which answers two
things at once:

  1. "You only benchmarked weak/old detectors."  It is a modern SSL system.
  2. "How much headroom is left at all?"  It is an approximate upper bound
     on what is reachable without target-language supervision.

It also previews the journal paper's full RQ3 without claiming it.

RELATIONSHIP TO STEPS 09/10 (journal track)
-------------------------------------------
Step 09 caches SSL features and Step 10 runs the full RQ3 transfer study
(adaptation curves, LOFO, typology regression). Those are the JOURNAL paper
and are disabled by default.

This step is the ICASSP deliverable and does one thing they do not: it emits
frame POSTERIORS in Step 04's npz layout under the system name `ssl_probe`,
so the probe flows into `frame_metrics.csv`, the variance decomposition and
the systems table with no downstream changes.

If Step 09's cache exists (`04_posteriors/ssl/index.csv`) this step reuses it
for training instead of re-encoding. Run 09 first if you want both tracks.

COST CONTROL
------------
SSL features are ~35 GB across the full 13-language grid, so inference-time
features are never cached. Training uses a reduced condition set
(`train_conditions`) on DEV sessions only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vadlib.audio import load_resampled, n_frames_for
from vadlib.config import resolve_device
from vadlib.corrupt import corrupt, speech_mask_frames, speech_mask_samples
from vadlib.decode import resample_posterior

FRAME_MS = 10
SYSTEM_NAME = "ssl_probe"


# --------------------------------------------------------------------------- #
class FrozenSSL:
    """Frozen SSL encoder; returns hidden states from one layer."""

    def __init__(self, model_name, layer, device, batch_s=30.0):
        self.model_name = model_name
        self.layer = int(layer)
        self.device = device
        self.batch_s = float(batch_s)
        self._m = None

    def load(self):
        import torch
        from transformers import AutoModel
        self._m = AutoModel.from_pretrained(self.model_name,
                                            output_hidden_states=True)
        self._m.eval().to(torch.device(self.device))
        for p in self._m.parameters():
            p.requires_grad_(False)
        return self

    @property
    def dim(self):
        return int(self._m.config.hidden_size)

    def features(self, x, sr):
        """(T_ssl, D) float32 hidden states from the configured layer."""
        import torch
        if self._m is None:
            self.load()
        dev = torch.device(self.device)
        chunk = int(self.batch_s * sr)
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), chunk):
                seg = x[i:i + chunk]
                if len(seg) < sr // 2:          # too short for a conv stack
                    continue
                t = torch.from_numpy(np.ascontiguousarray(seg,
                                                          dtype=np.float32))[None].to(dev)
                hs = self._m(t).hidden_states
                idx = min(self.layer, len(hs) - 1)
                outs.append(hs[idx][0].detach().cpu().numpy().astype(np.float32))
        if not outs:
            return np.zeros((1, self.dim), dtype=np.float32)
        return np.concatenate(outs, axis=0)


def _head(dim, hidden, device):
    import torch.nn as nn
    import torch
    return nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.ReLU(),
        nn.Dropout(0.1), nn.Linear(hidden, 1)).to(torch.device(device))


def _align(feats, n_ref):
    """SSL runs at 50 Hz, scoring at 100 Hz. Align labels to the feature rate."""
    return np.linspace(0, n_ref - 1, len(feats)).round().astype(int)


# --------------------------------------------------------------------------- #
def _try_step09_cache(cfg, sessions, train_conds):
    """Reuse Step 09's cached features if they cover the training conditions."""
    idx_p = Path(cfg["paths"]["post"]) / "ssl" / "index.csv"
    if not idx_p.exists():
        return None
    idx = pd.read_csv(idx_p)
    want = {c["cond_id"] for c in train_conds}
    have = set(idx["cond_id"].astype(str))
    usable = idx[idx["cond_id"].astype(str).isin(want & have)]
    dev_ids = set(sessions.loc[sessions["split"] == "dev", "session_id"])
    usable = usable[usable["session_id"].isin(dev_ids)]
    if usable.empty:
        return None
    X, Y, L = [], [], []
    for _, r in usable.iterrows():
        try:
            with np.load(r["path"]) as z:
                X.append(np.asarray(z["feat"], dtype=np.float32))
                y = np.asarray(z["label"], dtype=np.float32)
        except Exception:
            continue
        Y.append(y)
        L.extend([r["lang"]] * len(y))
    if not X:
        return None
    print(f"[step04b] reusing Step 09 cache: {len(usable)} feature files "
          f"({len(L)} frames) -- skipping re-encode for training")
    return np.concatenate(X), np.concatenate(Y), np.array(L)


def _collect_training(cfg, ssl, sessions, train_conds, sr):
    """Extract features + labels for DEV sessions over the reduced grid."""
    X, Y, L = [], [], []
    dev = sessions[sessions["split"] == "dev"]
    for _, s in dev.iterrows():
        with open(s["labels"]) as f:
            meta = json.load(f)
        clean = load_resampled(s["wav"], sr)
        smask = speech_mask_samples(meta["speech_intervals"], len(clean), sr)
        for cond in train_conds:
            x = corrupt(clean, sr, smask, cond, int(s["session_idx"]), cfg)
            f_ = ssl.features(x, sr)
            n_ref = n_frames_for(len(x), sr, 25, FRAME_MS)
            y_full = speech_mask_frames(meta["speech_intervals"], n_ref, FRAME_MS)
            y = y_full[_align(f_, n_ref)].astype(np.float32)
            X.append(f_)
            Y.append(y)
            L.extend([s["lang"]] * len(y))
    if not X:
        return None, None, None
    return np.concatenate(X), np.concatenate(Y), np.array(L)


def _train_head(X, Y, cfg, device):
    import torch
    import torch.nn as nn
    sp = cfg["ssl_probe"]
    model = _head(X.shape[1], int(sp["hidden"]), device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(sp["lr"]),
                            weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    dev = torch.device(device)
    Xt = torch.from_numpy(X).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    bs = int(sp["batch_size"])
    model.train()
    for _ in range(int(sp["epochs"])):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]).squeeze(-1), Yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


def run(cfg):
    sp = cfg.get("ssl_probe", {})
    if not sp.get("enabled", False):
        print("[step04b] disabled (ssl_probe.enabled: false) -- skipping")
        return None
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:
        print(f"[step04b] torch/transformers unavailable ({e}) -- skipping")
        return None

    sr = int(cfg["data"]["target_sr"])
    bench = Path(cfg["paths"]["bench"])
    post = Path(cfg["paths"]["post"])
    device = resolve_device(cfg)
    sessions = pd.read_csv(bench / "sessions.csv")
    conditions = pd.read_csv(bench / "conditions.csv").to_dict("records")

    train_ids = set(str(c) for c in sp.get("train_conditions", ["clean"]))
    train_conds = [c for c in conditions if str(c["cond_id"]) in train_ids]
    if not train_conds:
        train_conds = [conditions[0]]

    print(f"[step04b] device={device}  encoder={sp['model_name']} "
          f"layer={sp['layer']}")
    print(f"[step04b] training conditions: {[c['cond_id'] for c in train_conds]}")
    if device != "cuda":
        print("[step04b] WARNING: no GPU. SSL feature extraction on CPU over "
              "the full grid will take many hours. Consider ssl_probe.enabled: "
              "false, or reduce sessions/conditions.")

    ssl = FrozenSSL(sp["model_name"], sp["layer"], device,
                    float(sp.get("chunk_s", 30.0))).load()

    t0 = time.time()
    cached = _try_step09_cache(cfg, sessions, train_conds)
    if cached is not None:
        X, Y, L = cached
    else:
        print("[step04b] extracting training features ...")
        X, Y, L = _collect_training(cfg, ssl, sessions, train_conds, sr)
    if X is None:
        raise SystemExit("[step04b] no dev sessions -- check benchmark.dev_fraction")
    print(f"[step04b] {len(Y)} training frames, dim={X.shape[1]}, "
          f"{(time.time()-t0)/60:.1f} min")

    langs = sorted(sessions["lang"].unique())
    heads = {}
    for held in langs:
        tr = L != held
        if tr.sum() < 1000:
            print(f"[step04b] {held}: too little held-in data; skipping")
            continue
        h_t0 = time.time()
        heads[held] = _train_head(X[tr], Y[tr], cfg, device)
        print(f"[step04b] head excluding {held}: trained on "
              f"{int(tr.sum())} frames in {time.time()-h_t0:.0f}s")
    del X, Y, L

    # ----- inference over the full grid, TEST sessions only ------------------
    import torch
    test = sessions[sessions["split"] == "test"]
    out_dir = post / SYSTEM_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in conditions:
        out_npz = out_dir / f"{cond['cond_id']}.npz"
        if out_npz.exists() and not cfg["runtime"].get("overwrite", False):
            print(f"  [skip] {cond['cond_id']}")
            continue
        store, c_t0 = {}, time.time()
        for _, s in test.iterrows():
            head = heads.get(s["lang"])
            if head is None:
                continue
            with open(s["labels"]) as f:
                meta = json.load(f)
            clean = load_resampled(s["wav"], sr)
            smask = speech_mask_samples(meta["speech_intervals"], len(clean), sr)
            x = corrupt(clean, sr, smask, cond, int(s["session_idx"]), cfg)
            f_ = ssl.features(x, sr)
            with torch.no_grad():
                p = torch.sigmoid(
                    head(torch.from_numpy(f_).to(torch.device(device))
                         ).squeeze(-1)).cpu().numpy()
            n_ref = n_frames_for(len(x), sr, 25, FRAME_MS)
            store[s["session_id"]] = resample_posterior(p, n_ref).astype(np.float16)
        np.savez_compressed(out_npz, **store)
        dur = time.time() - c_t0
        audio_s = len(test) * sessions["duration_s"].mean()
        print(f"  [done] {cond['cond_id']:<28s} {dur:6.1f}s "
              f"({audio_s/max(dur,1e-6):6.1f}x realtime)")

    meta_out = Path(cfg["paths"]["results"]) / "ssl_probe_meta.csv"
    pd.DataFrame([{
        "system": SYSTEM_NAME, "encoder": sp["model_name"],
        "layer": int(sp["layer"]), "hidden": int(sp["hidden"]),
        "epochs": int(sp["epochs"]),
        "train_conditions": ";".join(c["cond_id"] for c in train_conds),
        "protocol": "leave-one-language-out",
        "n_heads": len(heads), "device": device,
        "category": "supervised in-domain, unseen target language",
    }]).to_csv(meta_out, index=False)

    print(f"\n[step04b] total {(time.time()-t0)/60:.1f} min -> {out_dir}")
    print("[step04b] REPORT THIS AS A SEPARATE CATEGORY in the systems table, "
          "not as a peer of the zero-shot systems.")
    return out_dir

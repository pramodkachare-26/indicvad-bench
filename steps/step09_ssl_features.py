"""Step 09 -- frozen SSL features + layer-wise probe.  RQ3, part 1.

THE ECONOMICS THAT MAKE THIS AFFORDABLE
---------------------------------------
The SSL encoder is FROZEN. Features are extracted once and cached to disk, so
every downstream experiment in Step 10 -- adaptation curves, leave-one-language
out, leave-one-family-out -- trains a small head on cached tensors and costs
seconds. That is the entire reason full RQ3 fits on free Colab: the expensive
part happens once.

Full LoRA/adapter fine-tuning of the encoder was considered and rejected:
8 languages x 5 budgets x 3 protocols is ~120 encoder training runs, roughly
20+ GPU-hours, which exceeds a free Colab allocation and would not change the
qualitative conclusion.

Layer-wise probe
----------------
VAD-relevant information in SSL models is known to peak in early-to-middle
layers rather than the final one, so taking the last hidden state -- the
default many papers use without checking -- is usually the wrong choice. The
probe measures this directly on a small subset and picks the layer used for
everything downstream. Report it; it is cheap and it justifies the choice.

Outputs
-------
  <post>/ssl/<cond_id>/<session_id>.npz    cached features, fp16
  ssl_layer_probe.csv                      AUC per layer (the probe result)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vadlib.audio import load_resampled
from vadlib.corrupt import corrupt, speech_mask_frames, speech_mask_samples

# SSL models emit one frame per 20 ms; the scoring grid is 10 ms. Labels are
# built at the SSL rate here and posteriors are upsampled at scoring time.
SSL_FRAME_MS = 20


def _load_encoder(cfg, device):
    from transformers import AutoModel
    name = cfg["ssl"]["model"]
    print(f"[step09] loading {name} on {device}")
    model = AutoModel.from_pretrained(name)
    model.eval()
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _encode(model, x, sr, device, layer=None, all_layers=False,
            chunk_s=10.0, overlap_s=0.5):
    """Encode a waveform to (T, D) features, or (L, T, D) if all_layers.

    Long audio is chunked to bound GPU memory; chunks overlap and the overlap
    is trimmed so boundary frames are never taken from a chunk edge.
    """
    import torch
    hop = int(chunk_s * sr)
    ov = int(overlap_s * sr)
    outs = []
    pos = 0
    while pos < len(x):
        lo = max(0, pos - ov)
        hi = min(len(x), pos + hop + ov)
        seg = np.ascontiguousarray(x[lo:hi], dtype=np.float32)
        with torch.no_grad():
            t = torch.from_numpy(seg)[None].to(device)
            out = model(t, output_hidden_states=all_layers or layer is not None)
            if all_layers:
                h = torch.stack(out.hidden_states, dim=0)[:, 0]      # (L, T, D)
            elif layer is not None:
                h = out.hidden_states[int(layer)][0][None]           # (1, T, D)
            else:
                h = out.last_hidden_state[0][None]
            h = h.float().cpu().numpy()
        # Trim the overlap in frames (20 ms per frame).
        fpad_lo = int(round((pos - lo) / sr * 1000 / SSL_FRAME_MS))
        want = int(round(min(hop, len(x) - pos) / sr * 1000 / SSL_FRAME_MS))
        outs.append(h[:, fpad_lo:fpad_lo + want])
        pos += hop
    feats = np.concatenate(outs, axis=1)
    return feats if all_layers else feats[0]


def _conditions_for_ssl(cfg):
    """RQ3 uses a condition SUBSET; the full 37-cell grid is unnecessary here
    and would inflate cache size ~10x for no additional claim."""
    bench = Path(cfg["paths"]["bench"])
    conds = pd.read_csv(bench / "conditions.csv").to_dict("records")
    keep_snr = set(cfg["ssl"]["conditions"]["snr_db"])
    keep_noise = set(cfg["ssl"]["conditions"]["noise_types"])
    keep_rev = set(cfg["ssl"]["conditions"]["reverb"])
    out = [c for c in conds
           if c["snr_db"] in keep_snr and c["noise"] in keep_noise
           and c["reverb"] in keep_rev]
    if cfg["ssl"]["conditions"].get("include_clean", True):
        clean = [c for c in conds if c["cond_id"] == "clean"]
        out = clean + [c for c in out if c["cond_id"] != "clean"]
    return out


def _labels_at_ssl_rate(intervals, n_frames):
    return speech_mask_frames(intervals, n_frames, SSL_FRAME_MS)


# --------------------------------------------------------------------------- #
def layer_probe(cfg, model, device, sessions, conds):
    """Which encoder layer carries the most VAD information?"""
    from sklearn.linear_model import LogisticRegression
    from vadlib.metrics import roc_auc

    sr = int(cfg["data"]["target_sr"])
    n_sess = int(cfg["ssl"]["probe_sessions_per_lang"])
    probe_conds = conds[: min(2, len(conds))]

    X_by_layer, Y = None, []
    picked = (sessions.sort_values("session_idx")
              .groupby("lang", group_keys=False).head(n_sess))
    print(f"[step09] layer probe on {len(picked)} sessions x "
          f"{len(probe_conds)} conditions")

    for _, s in picked.iterrows():
        with open(s["labels"]) as f:
            meta = json.load(f)
        clean = load_resampled(s["wav"], sr)
        smask = speech_mask_samples(meta["speech_intervals"], len(clean), sr)
        for cond in probe_conds:
            x = corrupt(clean, sr, smask, cond, int(s["session_idx"]), cfg)
            H = _encode(model, x, sr, device, all_layers=True)     # (L, T, D)
            y = _labels_at_ssl_rate(meta["speech_intervals"], H.shape[1])
            if X_by_layer is None:
                X_by_layer = [[] for _ in range(H.shape[0])]
            for li in range(H.shape[0]):
                X_by_layer[li].append(H[li].astype(np.float32))
            Y.append(y)

    Y = np.concatenate(Y)
    rows = []
    n = len(Y)
    idx = np.arange(n)
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    rng.shuffle(idx)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]

    for li, chunks in enumerate(X_by_layer):
        Xl = np.concatenate(chunks)
        mu, sd = Xl[tr].mean(0), Xl[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=400, n_jobs=-1)
        clf.fit((Xl[tr] - mu) / sd, Y[tr])
        p = clf.predict_proba((Xl[te] - mu) / sd)[:, 1]
        auc = roc_auc(Y[te], p)
        rows.append({"layer": li, "auc": auc, "n_train": len(tr),
                     "n_test": len(te)})
        print(f"  layer {li:2d}: AUC = {auc:.4f}")

    df = pd.DataFrame(rows)
    best = int(df.loc[df.auc.idxmax(), "layer"])
    df["is_best"] = df.layer == best
    df["model"] = cfg["ssl"]["model"]
    out = Path(cfg["paths"]["results"]) / "ssl_layer_probe.csv"
    df.to_csv(out, index=False)
    print(f"[step09] best layer = {best} (AUC {df.auc.max():.4f}); "
          f"last layer AUC = {df.auc.iloc[-1]:.4f}")
    return best, df


# --------------------------------------------------------------------------- #
def run(cfg):
    ssl_cfg = cfg.get("ssl", {})
    if not ssl_cfg.get("enabled", False):
        print("[step09] disabled (ssl.enabled: false) -- skipping")
        return None
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        print("[step09] needs torch + transformers -- skipping. "
              "pip install transformers")
        return None

    from vadlib.config import resolve_device
    device = resolve_device(cfg)
    if device != "cuda":
        print("[step09] WARNING: no GPU. Feature extraction on CPU is ~10x "
              "slower (hours, not minutes). Reduce ssl.conditions or set "
              "ssl.enabled: false.")

    sr = int(cfg["data"]["target_sr"])
    bench = Path(cfg["paths"]["bench"])
    post = Path(cfg["paths"]["post"])
    sessions = pd.read_csv(bench / "sessions.csv")
    conds = _conditions_for_ssl(cfg)
    print(f"[step09] {len(sessions)} sessions x {len(conds)} conditions "
          f"= {len(sessions)*len(conds)*sessions.duration_s.mean()/3600:.1f} h")

    model = _load_encoder(cfg, device)

    # --- 1. layer probe decides which layer to cache ------------------------
    layer = ssl_cfg.get("layer", "auto")
    probe_df = None
    if layer == "auto":
        layer, probe_df = layer_probe(cfg, model, device, sessions, conds)
    else:
        layer = int(layer)
        print(f"[step09] using configured layer {layer} (no probe)")

    # --- 2. cache features at that layer ------------------------------------
    meta_rows = []
    for cond in conds:
        out_dir = post / "ssl" / cond["cond_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        done = out_dir / ".complete"
        if done.exists() and not cfg["runtime"].get("overwrite", False):
            print(f"  [skip] {cond['cond_id']}")
        else:
            import time
            t0 = time.time()
            for _, s in sessions.iterrows():
                with open(s["labels"]) as f:
                    m = json.load(f)
                clean = load_resampled(s["wav"], sr)
                smask = speech_mask_samples(m["speech_intervals"], len(clean), sr)
                x = corrupt(clean, sr, smask, cond, int(s["session_idx"]), cfg)
                H = _encode(model, x, sr, device, layer=layer)
                y = _labels_at_ssl_rate(m["speech_intervals"], H.shape[0])
                np.savez_compressed(out_dir / f"{s['session_id']}.npz",
                                    feat=H.astype(np.float16),
                                    label=y.astype(np.uint8))
            done.touch()
            dt = time.time() - t0
            audio_s = len(sessions) * sessions.duration_s.mean()
            print(f"  [done] {cond['cond_id']:<26s} {dt:6.1f}s "
                  f"({audio_s/max(dt,1e-6):5.1f}x realtime)")
        for _, s in sessions.iterrows():
            meta_rows.append({"session_id": s["session_id"], "lang": s["lang"],
                              "family": s["family"], "split": s["split"],
                              "session_idx": int(s["session_idx"]),
                              "cond_id": cond["cond_id"], "snr_db": cond["snr_db"],
                              "noise": cond["noise"], "reverb": cond["reverb"],
                              "path": str(post / "ssl" / cond["cond_id"] /
                                          f"{s['session_id']}.npz")})

    idx = pd.DataFrame(meta_rows)
    idx["layer"] = layer
    idx["model"] = ssl_cfg["model"]
    idx.to_csv(post / "ssl" / "index.csv", index=False)
    print(f"[step09] cached {len(idx)} feature files at layer {layer}")
    return post / "ssl" / "index.csv"

"""Step 04 -- inference. Runs each enabled VAD over the full condition grid.

Posteriors (not decisions) are cached to disk as float16 npz. This is what
makes Step 07 -- the decoding-hyperparameter sweep, and the headline result --
cost seconds rather than re-running inference thousands of times.

This is the dominant cost of the pipeline. Progress is checkpointed per
(system, condition), so an interrupted run resumes without loss.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vadlib.audio import load_resampled, n_frames_for
from vadlib.corrupt import corrupt, speech_mask_samples
from vadlib.config import resolve_device
from vadlib.systems import REGISTRY, build_system, enabled_systems

FRAME_MS = 10


def _select_sessions(sessions, subsample, seed=0):
    if subsample >= 1.0:
        return sessions
    rng = np.random.default_rng(seed)
    keep = max(1, int(round(len(sessions) * subsample)))
    idx = rng.choice(len(sessions), size=keep, replace=False)
    return sessions.iloc[np.sort(idx)]


def run(cfg):
    sr = int(cfg["data"]["target_sr"])
    bench = Path(cfg["paths"]["bench"])
    post = Path(cfg["paths"]["post"])
    sessions = pd.read_csv(bench / "sessions.csv")
    conditions = pd.read_csv(bench / "conditions.csv").to_dict("records")
    systems = enabled_systems(cfg)
    if not systems:
        raise SystemExit("[step04] no systems enabled in config.yaml")

    device = resolve_device(cfg)
    print(f"[step04] device={device}  systems={systems}  "
          f"sessions={len(sessions)}  conditions={len(conditions)}")

    for sys_name in systems:
        spec = cfg["systems"][sys_name]
        sub = float(spec.get("subsample", 1.0))
        sess = _select_sessions(sessions, sub, seed=int(cfg["project"]["seed"]))
        print(f"\n[step04] === {sys_name} ({len(sess)} sessions x "
              f"{len(conditions)} conditions) ===")
        on_gpu = device == "cuda" and getattr(REGISTRY.get(sys_name),
                                              "uses_gpu", False)
        print(f"[step04] running on {'cuda' if on_gpu else 'cpu'}")
        try:
            model = build_system(sys_name, cfg, device=device).load()
        except Exception as e:
            print(f"[step04] SKIP {sys_name}: could not load ({e})")
            continue

        for cond in conditions:
            out_npz = post / sys_name / f"{cond['cond_id']}.npz"
            if out_npz.exists() and not cfg["runtime"].get("overwrite", False):
                print(f"  [skip] {cond['cond_id']}")
                continue
            out_npz.parent.mkdir(parents=True, exist_ok=True)

            store, t0 = {}, time.time()
            for _, s in sess.iterrows():
                with open(s["labels"]) as f:
                    ivs = json.load(f)["speech_intervals"]
                clean = load_resampled(s["wav"], sr)
                mask = speech_mask_samples(ivs, len(clean), sr)
                x = corrupt(clean, sr, mask, cond, int(s["session_idx"]), cfg)
                try:
                    p = model.posterior(x, sr)
                except Exception as e:
                    print(f"    [warn] {sys_name}/{s['session_id']}: {e}")
                    p = np.zeros(n_frames_for(len(x), sr, 25, FRAME_MS),
                                 dtype=np.float32)
                store[s["session_id"]] = np.asarray(p, dtype=np.float16)

            np.savez_compressed(out_npz, **store)
            dur = time.time() - t0
            audio_s = len(sess) * sessions["duration_s"].mean()
            rtf = audio_s / max(dur, 1e-6)
            remaining = (len(conditions) - conditions.index(cond) - 1) * dur / 60
            print(f"  [done] {cond['cond_id']:<28s} {dur:6.1f}s "
                  f"({rtf:6.1f}x realtime)  ~{remaining:.0f} min left "
                  f"for {sys_name}")

    print(f"\n[step04] posteriors cached under {post}")
    return post

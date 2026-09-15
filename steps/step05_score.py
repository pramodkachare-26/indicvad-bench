"""Step 05 -- score cached posteriors -> results/frame_metrics.csv

One row per (system, session, condition). This long-format table is the input
to every downstream analysis and to the paper's tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vadlib.corrupt import speech_mask_frames
from vadlib.decode import decode
from vadlib.metrics import score_all

FRAME_MS = 10


def run(cfg):
    bench = Path(cfg["paths"]["bench"])
    post = Path(cfg["paths"]["post"])
    res = Path(cfg["paths"]["results"])
    sc = cfg["scoring"]

    sessions = pd.read_csv(bench / "sessions.csv").set_index("session_id")
    conditions = pd.read_csv(bench / "conditions.csv").set_index("cond_id")

    refs = {}
    for sid, row in sessions.iterrows():
        with open(row["labels"]) as f:
            meta = json.load(f)
        n_fr = int(round(meta["duration_s"] / (FRAME_MS / 1000.0)))
        refs[sid] = speech_mask_frames(meta["speech_intervals"], n_fr, FRAME_MS)

    rows = []
    for sys_dir in sorted(post.glob("*")):
        if not sys_dir.is_dir():
            continue
        sys_name = sys_dir.name
        for npz_path in sorted(sys_dir.glob("*.npz")):
            cond_id = npz_path.stem
            if cond_id not in conditions.index:
                continue
            cond = conditions.loc[cond_id]
            with np.load(npz_path) as z:
                for sid in z.files:
                    if sid not in refs:
                        continue
                    ref = refs[sid]
                    p = np.asarray(z[sid], dtype=np.float32)
                    if len(p) < len(ref):
                        p = np.pad(p, (0, len(ref) - len(p)))
                    p = p[: len(ref)]
                    hyp = decode(p, FRAME_MS, threshold=0.5,
                                 min_speech_s=0.0, min_silence_s=0.0, pad_s=0.0)
                    m = score_all(
                        ref, p, hyp, FRAME_MS,
                        collars=tuple(sc["collars_s"]),
                        tolerances=tuple(sc["boundary_tolerances_s"]),
                        match_window_s=float(sc["boundary_match_window_s"]),
                        w_miss=float(sc["dcf_miss_weight"]),
                        w_fa=float(sc["dcf_fa_weight"]))
                    s = sessions.loc[sid]
                    rows.append({
                        "system": sys_name, "session_id": sid,
                        "lang": s["lang"], "family": s["family"],
                        "session_idx": int(s["session_idx"]), "split": s["split"],
                        "cond_id": cond_id, "snr_db": int(cond["snr_db"]),
                        "noise": cond["noise"], "reverb": cond["reverb"],
                        "ref_speech_ratio": float(ref.mean()),
                        **m})
            print(f"  [scored] {sys_name}/{cond_id}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[step05] nothing scored -- did step04 produce npz files?")
    out = res / "frame_metrics.csv"
    df.to_csv(out, index=False)

    print(f"\n[step05] {len(df)} rows -> {out}")
    piv = df[df.snr_db < 99].pivot_table(index="lang", columns="system",
                                         values="deter", aggfunc="mean")
    print("\nMean DetER by language x system (noisy conditions):")
    print(piv.round(4).to_string())
    return out

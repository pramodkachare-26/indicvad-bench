"""Step 07 -- does 'language dependence' live in three decoding scalars?

Fits (threshold, min_speech, min_silence, pad) on DEV sessions and evaluates on
TEST sessions, three ways:

  global     one setting shared by all languages           (the usual practice)
  perlang    one setting per language                      (the 'adapted' upper bound)
  oracle     per-language setting fitted on test           (headroom check)

The reported quantity is GAP CLOSURE: the fraction of the global-to-oracle gap
recovered by per-language retuning alone, with zero retraining and no target
audio beyond a few minutes of dev. If this is large, the practical answer to
RQ3 is 'retune, don't retrain' -- which is a positive, actionable finding even
if the language main effect in Step 06 is null.

Runs on cached posteriors: seconds, not hours.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vadlib.corrupt import speech_mask_frames
from vadlib.decode import decode
from vadlib.metrics import detection_metrics

FRAME_MS = 10


def _load_refs(cfg, sessions):
    refs = {}
    for sid, row in sessions.iterrows():
        with open(row["labels"]) as f:
            meta = json.load(f)
        n_fr = int(round(meta["duration_s"] / (FRAME_MS / 1000.0)))
        refs[sid] = speech_mask_frames(meta["speech_intervals"], n_fr, FRAME_MS)
    return refs


def _load_posteriors(post, sys_name):
    store = {}
    for npz_path in sorted((post / sys_name).glob("*.npz")):
        with np.load(npz_path) as z:
            for sid in z.files:
                store[(npz_path.stem, sid)] = np.asarray(z[sid], dtype=np.float32)
    return store


def _grid(cfg):
    g = cfg["hparam_grid"]
    return list(itertools.product(g["threshold"], g["min_speech_s"],
                                  g["min_silence_s"], g["pad_s"]))


def _eval_setting(keys, store, refs, setting, w_miss, w_fa):
    thr, msp, msi, pad = setting
    tot_err, tot_sp = 0.0, 0.0
    for k in keys:
        p = store.get(k)
        if p is None:
            continue
        ref = refs[k[1]]
        if len(p) < len(ref):
            p = np.pad(p, (0, len(ref) - len(p)))
        p = p[: len(ref)]
        hyp = decode(p, FRAME_MS, thr, msp, msi, pad)
        d = detection_metrics(ref, hyp, FRAME_MS, 0.0, w_miss, w_fa)
        tot_err += d["miss_s"] + d["fa_s"]
        tot_sp += d["n_speech_s"]
    return tot_err / max(tot_sp, 1e-9)


def run(cfg):
    bench = Path(cfg["paths"]["bench"])
    post = Path(cfg["paths"]["post"])
    res = Path(cfg["paths"]["results"])
    sc = cfg["scoring"]
    w_miss, w_fa = float(sc["dcf_miss_weight"]), float(sc["dcf_fa_weight"])

    sessions = pd.read_csv(bench / "sessions.csv").set_index("session_id")
    refs = _load_refs(cfg, sessions)
    grid = _grid(cfg)
    print(f"[step07] grid size = {len(grid)} settings")

    langs = sorted(sessions["lang"].unique())
    dev_ids = set(sessions.index[sessions["split"] == "dev"])
    test_ids = set(sessions.index[sessions["split"] == "test"])

    rows, best_rows = [], []
    for sys_dir in sorted(post.glob("*")):
        if not sys_dir.is_dir():
            continue
        sys_name = sys_dir.name
        store = _load_posteriors(post, sys_name)
        if not store:
            continue
        by_lang_dev = {L: [k for k in store
                           if k[1] in dev_ids and sessions.loc[k[1], "lang"] == L]
                       for L in langs}
        by_lang_test = {L: [k for k in store
                            if k[1] in test_ids and sessions.loc[k[1], "lang"] == L]
                        for L in langs}
        all_dev = [k for k in store if k[1] in dev_ids]

        # --- global setting fitted on pooled dev -----------------------------
        gscores = [_eval_setting(all_dev, store, refs, s, w_miss, w_fa)
                   for s in grid]
        g_best = grid[int(np.argmin(gscores))]
        print(f"[step07] {sys_name}: global best = thr={g_best[0]} "
              f"minsp={g_best[1]} minsil={g_best[2]} pad={g_best[3]}")

        for L in langs:
            if not by_lang_test[L]:
                continue
            n_dev_sess = len({k[1] for k in by_lang_dev[L]})
            if n_dev_sess < 3:
                print(f"  [warn] {L}: only {n_dev_sess} dev sessions -- "
                      "per-language tuning will overfit and may look WORSE "
                      "than global. Raise benchmark.sessions_per_lang.")
            # per-language, fitted on that language's dev split
            dsc = [_eval_setting(by_lang_dev[L], store, refs, s, w_miss, w_fa)
                   for s in grid] if by_lang_dev[L] else gscores
            l_best = grid[int(np.argmin(dsc))]
            # oracle, fitted on test (headroom only -- never reported as a result)
            tsc = [_eval_setting(by_lang_test[L], store, refs, s, w_miss, w_fa)
                   for s in grid]
            o_best = grid[int(np.argmin(tsc))]

            e_glob = _eval_setting(by_lang_test[L], store, refs, g_best, w_miss, w_fa)
            e_lang = _eval_setting(by_lang_test[L], store, refs, l_best, w_miss, w_fa)
            e_orac = float(np.min(tsc))
            gap = e_glob - e_orac
            rows.append({
                "system": sys_name, "lang": L,
                "family": sessions.loc[by_lang_test[L][0][1], "family"],
                "deter_global": e_glob, "deter_perlang": e_lang,
                "deter_oracle": e_orac,
                "gap_global_oracle": gap,
                "gap_closed_frac": (e_glob - e_lang) / gap if gap > 1e-9 else np.nan,
                "rel_improvement": (e_glob - e_lang) / max(e_glob, 1e-9),
                "n_dev_sessions": n_dev_sess,
                "overfit_flag": bool(e_lang > e_glob),
                "global_threshold": g_best[0], "global_min_speech_s": g_best[1],
                "global_min_silence_s": g_best[2], "global_pad_s": g_best[3],
                "best_threshold": l_best[0], "best_min_speech_s": l_best[1],
                "best_min_silence_s": l_best[2], "best_pad_s": l_best[3],
                "oracle_threshold": o_best[0], "oracle_min_silence_s": o_best[2],
            })
        best_rows.append({"system": sys_name, "global_threshold": g_best[0],
                          "global_min_speech_s": g_best[1],
                          "global_min_silence_s": g_best[2],
                          "global_pad_s": g_best[3],
                          "dev_deter": float(np.min(gscores))})

    df = pd.DataFrame(rows)
    df.to_csv(res / "hparam_tuning.csv", index=False)
    pd.DataFrame(best_rows).to_csv(res / "hparam_global_settings.csv", index=False)

    if not df.empty:
        print("\n[step07] Per-language retuning, relative DetER improvement:")
        print(df.pivot_table(index="lang", columns="system",
                             values="rel_improvement").round(4).to_string())
        print("\n[step07] Fraction of global->oracle gap closed by retuning alone:")
        print(df.groupby("system")["gap_closed_frac"].mean().round(3).to_string())
        print("\n[step07] Does the optimal min-silence shift by language? "
              "(H3: syllable-timed rhythm)")
        print(df.pivot_table(index="lang", columns="system",
                             values="best_min_silence_s").to_string())
    print(f"\n[step07] wrote hparam_tuning.csv to {res}")
    return res / "hparam_tuning.csv"

"""Step 03 -- synthesize the benchmark. Ground truth is exact by construction.

THE CRITICAL DESIGN CONSTRAINT
------------------------------
Silence durations, excerpt ordering seeds, noise-clip choices and RIR choices
are all seeded on SESSION INDEX ONLY -- never on language. Session 7 in Hindi
and session 7 in Tamil therefore receive the identical pause structure, the
identical noise waveform and the identical room. Combined with FLEURS's
parallel FLoRes content, this leaves language as the sole free factor.

Without this, any measured "language effect" is a corpus effect, and the whole
research question collapses. Do not remove the shared seeding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vadlib.audio import load_resampled, write_wav
from vadlib.config import control_languages, test_languages


def _pause_durations(n, rng, cfg):
    p = cfg["benchmark"]["pause_lognormal"]
    d = rng.lognormal(mean=float(p["mu"]), sigma=float(p["sigma"]), size=n)
    return np.clip(d, float(p["min_s"]), float(p["max_s"]))


def build_session(excerpt_paths, sr, session_idx, cfg):
    """Return (waveform, speech_intervals, excerpt_ids_used)."""
    b = cfg["benchmark"]
    rng = np.random.default_rng(int(cfg["project"]["seed"]) + 1000 * session_idx)
    target = float(b["session_duration_s"])
    lead = float(b["lead_trail_sil_s"])

    pauses = _pause_durations(200, rng, cfg)
    chunks, intervals, t = [], [], 0.0
    chunks.append(np.zeros(int(lead * sr), dtype=np.float32))
    t += lead

    pi, used = 0, []
    for p in excerpt_paths:
        if t >= target - lead:
            break
        x = load_resampled(p, sr)
        dur = len(x) / sr
        if t + dur > target - lead:
            keep = int(max(0.0, target - lead - t) * sr)
            if keep < int(0.5 * sr):
                break
            x, dur = x[:keep], keep / sr
        chunks.append(x)
        intervals.append((t, t + dur))
        used.append(Path(p).stem)
        t += dur
        gap = float(pauses[pi % len(pauses)]); pi += 1
        if t + gap < target - lead:
            chunks.append(np.zeros(int(gap * sr), dtype=np.float32))
            t += gap

    chunks.append(np.zeros(int(lead * sr), dtype=np.float32))
    wav = np.concatenate(chunks).astype(np.float32)
    want = int(target * sr)
    wav = wav[:want] if len(wav) > want else np.pad(wav, (0, want - len(wav)))
    intervals = [(a, min(b_, len(wav) / sr)) for a, b_ in intervals if a < len(wav) / sr]
    return wav, intervals, used


def run(cfg):
    sr = int(cfg["data"]["target_sr"])
    bench = Path(cfg["paths"]["bench"])
    pool = pd.read_csv(Path(cfg["paths"]["pool"]) / "excerpts_all.csv")

    test_codes = [l["code"] for l in test_languages(cfg)]
    ctrl_codes = [l["code"] for l in control_languages(cfg)]

    n_sess = int(cfg["benchmark"]["sessions_per_lang"])
    dev_frac = float(cfg["benchmark"]["dev_fraction"])
    n_dev = int(round(n_sess * dev_frac))

    rows = []
    for lang in test_codes:
        sub = pool[pool["lang"] == lang]
        if sub.empty:
            print(f"[step03] WARNING: no excerpts for {lang}; skipping")
            continue
        out_dir = bench / lang
        out_dir.mkdir(parents=True, exist_ok=True)

        for si in range(n_sess):
            # Excerpt SELECTION may vary by language (different sentences
            # exist per language) but pause structure is shared -- see above.
            rng = np.random.default_rng(hash((lang, si)) % (2 ** 32))
            order = rng.permutation(len(sub))
            paths = [sub.iloc[int(i)]["path"] for i in order[:60]]

            wav, ivs, used = build_session(paths, sr, si, cfg)
            wpath = out_dir / f"{lang}_s{si:03d}.wav"
            write_wav(wpath, wav, sr)
            with open(out_dir / f"{lang}_s{si:03d}.json", "w") as f:
                json.dump({"speech_intervals": ivs, "sr": sr,
                           "duration_s": len(wav) / sr, "excerpts": used}, f)
            speech_s = sum(b - a for a, b in ivs)
            rows.append({
                "session_id": f"{lang}_s{si:03d}", "lang": lang,
                "family": sub.iloc[0]["family"], "session_idx": si,
                "split": "dev" if si < n_dev else "test",
                "wav": str(wpath), "labels": str(out_dir / f"{lang}_s{si:03d}.json"),
                "duration_s": len(wav) / sr, "speech_s": speech_s,
                "n_segments": len(ivs),
                "speech_ratio": speech_s / max(len(wav) / sr, 1e-9),
            })
        print(f"[step03] {lang}: {n_sess} sessions built")

    # Babble source: excerpts from CONTROL languages only, so babble never
    # contains a language under test.
    babble = pool[pool["lang"].isin(ctrl_codes)]["path"].tolist()
    if not babble:
        print("[step03] WARNING: no control-language excerpts; babble will be "
              "synthetic noise. Add a language with control: true.")
    with open(bench / "babble_sources.json", "w") as f:
        json.dump(babble[:400], f)

    df = pd.DataFrame(rows)
    out = bench / "sessions.csv"
    df.to_csv(out, index=False)

    conds = _condition_grid(cfg)
    pd.DataFrame(conds).to_csv(bench / "conditions.csv", index=False)

    print(f"[step03] {len(df)} sessions, {df['duration_s'].sum()/3600:.2f} h clean")
    print(f"[step03] {len(conds)} conditions -> "
          f"{len(df)*len(conds)*df['duration_s'].mean()/3600:.1f} h to score per system")
    print(df.groupby("lang")[["speech_ratio", "n_segments"]].mean().to_string())
    return out


def _condition_grid(cfg):
    c = cfg["conditions"]
    grid = []
    for snr in c["snr_db"]:
        for nz in c["noise_types"]:
            for rv in c["reverb"]:
                grid.append({"cond_id": f"{nz}_snr{snr}_{rv}",
                             "snr_db": snr, "noise": nz, "reverb": rv})
    # A clean anchor condition keeps the SNR axis interpretable.
    grid.insert(0, {"cond_id": "clean", "snr_db": 99, "noise": "none",
                    "reverb": "none"})
    return grid

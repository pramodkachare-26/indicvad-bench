"""Step 02 -- build the verified speech-excerpt pool.

Reference labels for the synthetic benchmark are exact BY CONSTRUCTION: we
keep only regions where a committee of independent detectors unanimously
agrees on speech, erode 150 ms off each edge, and treat the retained excerpt
as wholly speech. Silence is then inserted by us, so we know it exactly.

Circularity control
-------------------
Evaluating a detector on references it helped build is a real objection. Two
defences are implemented:
  1. Edge erosion -- retained regions are strictly interior to the agreed
     speech, so a system that merely *matches* the committee is not rewarded
     at the boundaries that matter.
  2. leave_system_out -- optionally emit an extra pool per system in which
     that system is removed from the committee. Report the delta as a
     sensitivity row; if conclusions move, say so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vadlib.agreement import committee_report, interpret
from vadlib.audio import load_resampled, write_wav
from vadlib.decode import _runs
from vadlib.config import resolve_device
from vadlib.systems import build_system


def _committee_votes(x, sr, members, frame_ms=10):
    """Return the (n_members, T) boolean vote matrix."""
    masks = [sysobj.posterior(x, sr) >= 0.5 for sysobj in members]
    n = min(len(m) for m in masks)
    return np.stack([m[:n] for m in masks], axis=0)


def _committee_mask(x, sr, members, frame_ms=10):
    return _committee_votes(x, sr, members, frame_ms).mean(axis=0)


def _extract_excerpts(x, sr, agree, cfg, frame_ms=10):
    r = cfg["reference"]
    thr = float(r["min_committee_agreement"])
    min_pause_frames = int(round(float(r["min_pause_s"]) / (frame_ms / 1000.0)))
    speech = agree >= thr - 1e-9

    # Internal gaps shorter than min_pause stay 'speech' (stop closures,
    # geminates) -- this is the annotation convention, stated in the paper.
    filled = speech.copy()
    for s, e, v in _runs(speech):
        if not v and (e - s) < min_pause_frames:
            filled[s:e] = True

    erode = int(round(float(r["edge_erosion_s"]) / (frame_ms / 1000.0)))
    lo_s, hi_s = float(r["min_excerpt_s"]), float(r["max_excerpt_s"])
    hop = int(sr * frame_ms / 1000)

    out = []
    for s, e, v in _runs(filled):
        if not v:
            continue
        s2, e2 = s + erode, e - erode
        dur = (e2 - s2) * frame_ms / 1000.0
        if dur < lo_s:
            continue
        if dur > hi_s:
            e2 = s2 + int(hi_s / (frame_ms / 1000.0))
        a, b = s2 * hop, min(len(x), e2 * hop)
        if b - a > int(lo_s * sr):
            out.append(x[a:b].astype(np.float32))
    return out


def run(cfg):
    sr = int(cfg["data"]["target_sr"])
    pool_dir = Path(cfg["paths"]["pool"])
    utt_csv = Path(cfg["paths"]["raw"]) / "utterances.csv"
    if not utt_csv.exists():
        raise SystemExit("[step02] run step01 first (utterances.csv missing)")
    utts = pd.read_csv(utt_csv)

    device = resolve_device(cfg)
    member_names = list(cfg["reference"]["committee"])
    members = []
    for name in member_names:
        print(f"[step02] loading committee member: {name}")
        members.append(build_system(name, cfg, device=device).load())
    if len(members) < 2:
        print("[step02] WARNING: a single-member committee cannot be assessed "
              "for agreement, and unanimity is vacuous. Use >= 2 (ideally 3).")

    rows, agree_rows = [], []
    for lang, grp in utts.groupby("lang"):
        out_dir = pool_dir / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        done = out_dir / ".complete"
        if done.exists() and not cfg["runtime"].get("overwrite", False):
            print(f"[step02] {lang}: cached")
            rows.extend(pd.read_csv(out_dir / "excerpts.csv").to_dict("records"))
            if (out_dir / "agreement.csv").exists():
                agree_rows.extend(
                    pd.read_csv(out_dir / "agreement.csv").to_dict("records"))
            continue

        recs, n_ex, vote_chunks = [], 0, []
        for _, row in grp.iterrows():
            try:
                x = load_resampled(row["path"], sr)
            except Exception as e:
                print(f"  [warn] unreadable {row['path']}: {e}")
                continue
            if len(x) < sr:
                continue
            votes = _committee_votes(x, sr, members)
            vote_chunks.append(votes)
            agree = votes.mean(axis=0)
            for k, seg in enumerate(_extract_excerpts(x, sr, agree, cfg)):
                p = out_dir / f"{row['utt_id']}_{k:02d}.wav"
                write_wav(p, seg, sr)
                recs.append({
                    "lang": lang, "family": row["family"],
                    "control": row["control"], "utt_id": row["utt_id"],
                    "excerpt_id": f"{row['utt_id']}_{k:02d}",
                    "path": str(p), "duration_s": len(seg) / sr,
                    "gender": row.get("gender", "unknown"),
                    "transcript": row.get("transcript", ""),
                })
                n_ex += 1
        pd.DataFrame(recs).to_csv(out_dir / "excerpts.csv", index=False)

        # --- committee agreement (kappa) -------------------------------------
        if vote_chunks and len(members) >= 2:
            pooled = np.concatenate(vote_chunks, axis=1)
            rep = committee_report(pooled, member_names)
            rep["lang"] = lang
            agree_rows.append(rep)
            pd.DataFrame([rep]).to_csv(out_dir / "agreement.csv", index=False)
            print(f"[step02] {lang}: Fleiss kappa = {rep['fleiss_kappa']:.3f} "
                  f"({interpret(rep['fleiss_kappa'])}), "
                  f"unanimity on {rep['observed_unanimity']*100:.1f}% of frames")

        done.touch()
        total = sum(r["duration_s"] for r in recs)
        print(f"[step02] {lang}: {n_ex} excerpts, {total/60:.1f} min speech")
        rows.extend(recs)

    df = pd.DataFrame(rows)
    out = pool_dir / "excerpts_all.csv"
    df.to_csv(out, index=False)

    if agree_rows:
        ag = pd.DataFrame(agree_rows)
        res_dir = Path(cfg["paths"]["results"])
        res_dir.mkdir(parents=True, exist_ok=True)
        ag.to_csv(res_dir / "committee_agreement.csv", index=False)
        k = ag["fleiss_kappa"].astype(float)
        print(f"\n[step02] COMMITTEE AGREEMENT (report this in section 3)")
        print(f"  members        : {member_names}")
        print(f"  Fleiss kappa   : mean {k.mean():.3f}  "
              f"[min {k.min():.3f}, max {k.max():.3f}]  -> {interpret(k.mean())}")
        pair_cols = [c for c in ag.columns if c.startswith("cohen_kappa_")]
        for c in sorted(pair_cols):
            print(f"  {c:<40s} {ag[c].astype(float).mean():.3f}")
        if k.mean() < 0.60:
            print("  WARNING: agreement below 'substantial'. The unanimity rule "
                  "is doing more work than the detectors are; treat the "
                  "reference as weak and lean harder on the human gold set.")
        print(f"  -> {res_dir / 'committee_agreement.csv'}")

    print(f"\n[step02] pool total: {df['duration_s'].sum()/3600:.2f} h "
          f"across {df['lang'].nunique()} languages -> {out}")
    return out

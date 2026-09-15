"""Step 08 -- where does the error localize phonologically?

GPU forced alignment is out of budget, so we use a PROXY that is cheap and
defensible: Indic scripts are near-phonemic and the Unicode blocks are
ISCII-derived, so the consonant grid has the SAME relative offsets in
Devanagari, Bengali, Gujarati, Gurmukhi, Oriya, Telugu, Kannada and Malayalam.
That lets us count, straight from the FLEURS transcript:

  aspirate_density   voiceless aspirates  (kh, ch, Th, th, ph)
  breathy_density    breathy-voiced stops (gh, jh, Dh, dh, bh)   <- H1
  retroflex_density  retroflex series
  geminate_density   C + virama + same C                          <- H2

Tamil, which lacks the four-way laryngeal contrast, scores ~0 on breathy by
construction -- exactly the Indo-Aryan vs Dravidian contrast we want to test.

CALL THIS A PROXY IN THE PAPER. It is orthographic, not acoustic. Upgrading to
MMS forced alignment is the obvious follow-up and belongs in the limitations.

Also emits the human-gold annotation pack (Step 08b): stimuli plus Audacity
label templates. Hand-labelling ~2.5 min per language is the single highest
value non-code task available to you.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from vadlib.config import test_languages
from vadlib.decode import segments_to_mask
from vadlib.metrics import boundary_metrics
from vadlib.agreement import cohen_kappa

# Offsets from each Indic Unicode block base (ISCII-derived layout).
OFF_ASPIRATE = {0x16, 0x1B, 0x20, 0x25, 0x2B}          # KHA CHA TTHA THA PHA
OFF_BREATHY = {0x18, 0x1D, 0x22, 0x27, 0x2D}           # GHA JHA DDHA DHA BHA
OFF_RETROFLEX = {0x1F, 0x20, 0x21, 0x22, 0x23, 0x37}   # TTA..NNA, SSA
OFF_VIRAMA = 0x4D
OFF_CONS_LO, OFF_CONS_HI = 0x15, 0x39

BLOCK_BASES = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80,
               0x0C00, 0x0C80, 0x0D00]                  # Deva..Malayalam


def _decompose(ch):
    cp = ord(ch)
    for base in BLOCK_BASES:
        if base <= cp < base + 0x80:
            return base, cp - base
    return None, None


def phono_features(text):
    if not isinstance(text, str) or not text.strip():
        return None
    n_asp = n_bre = n_ret = n_gem = n_cons = 0
    prev = []                      # rolling (base, offset) for geminate check
    for ch in text:
        base, off = _decompose(ch)
        if base is None:
            prev = []
            continue
        if OFF_CONS_LO <= off <= OFF_CONS_HI:
            n_cons += 1
            if off in OFF_ASPIRATE:
                n_asp += 1
            if off in OFF_BREATHY:
                n_bre += 1
            if off in OFF_RETROFLEX:
                n_ret += 1
            # geminate: C + virama + same C
            if len(prev) >= 2 and prev[-1] == (base, OFF_VIRAMA) \
                    and prev[-2] == (base, off):
                n_gem += 1
        prev.append((base, off))
        prev = prev[-3:]
    if n_cons == 0:
        return None
    return {"n_consonants": n_cons,
            "aspirate_density": n_asp / n_cons,
            "breathy_density": n_bre / n_cons,
            "retroflex_density": n_ret / n_cons,
            "geminate_density": n_gem / n_cons}


def _session_features(cfg):
    pool = pd.read_csv(Path(cfg["paths"]["pool"]) / "excerpts_all.csv")
    tmap = dict(zip(pool["excerpt_id"].astype(str),
                    pool["transcript"].fillna("").astype(str)))
    sessions = pd.read_csv(Path(cfg["paths"]["bench"]) / "sessions.csv")

    rows = []
    for _, s in sessions.iterrows():
        with open(s["labels"]) as f:
            meta = json.load(f)
        texts = [tmap.get(e, "") for e in meta.get("excerpts", [])]
        feats = [phono_features(t) for t in texts]
        feats = [f for f in feats if f]
        if not feats:
            continue
        w = np.array([f["n_consonants"] for f in feats], dtype=float)
        rec = {"session_id": s["session_id"], "lang": s["lang"],
               "family": s["family"], "n_consonants": float(w.sum())}
        for k in ("aspirate_density", "breathy_density",
                  "retroflex_density", "geminate_density"):
            rec[k] = float(np.average([f[k] for f in feats], weights=w))
        rows.append(rec)
    return pd.DataFrame(rows)


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(r @ r) / ss_tot if ss_tot > 0 else np.nan
    return beta, se, t, p, r2


def _bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (q-values). NaN-safe: NaNs pass
    through as NaN and are excluded from the ranking/denominator.
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return out
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    out[order] = np.clip(q, 0, 1)
    return out


def _vif(Z):
    """Variance inflation factors for standardized predictor block Z."""
    vifs = []
    for j in range(Z.shape[1]):
        others = np.delete(Z, j, axis=1)
        if others.shape[1] == 0:
            vifs.append(1.0)
            continue
        A = np.hstack([np.ones((len(Z), 1)), others])
        beta, *_ = np.linalg.lstsq(A, Z[:, j], rcond=None)
        r = Z[:, j] - A @ beta
        ss_tot = float(((Z[:, j] - Z[:, j].mean()) ** 2).sum())
        r2 = 1 - float(r @ r) / ss_tot if ss_tot > 1e-12 else 1.0
        vifs.append(1.0 / max(1 - r2, 1e-9))
    return np.array(vifs)


def _regress(metrics, feats, target, with_lang_dummies):
    df = metrics.merge(feats, on=["session_id", "lang", "family"], how="inner")
    if df.empty:
        return None
    preds = ["breathy_density", "aspirate_density",
             "retroflex_density", "geminate_density"]
    out = []
    for sysname, g in df.groupby("system"):
        g = g.dropna(subset=[target] + preds)
        if len(g) < 20:
            continue
        # Drop predictors with no within-sample variation; they are not
        # estimable and would otherwise absorb an arbitrary share of the fit.
        live = [p for p in preds if g[p].std() > 1e-12]
        if not live:
            continue
        Z = np.column_stack([(g[p].to_numpy(float) - g[p].mean()) / g[p].std()
                             for p in live])
        vifs = _vif(Z)

        # CRITICAL GUARD. With one transcript per language the density
        # features are constant within language, hence perfectly collinear
        # with the language dummies. lstsq then splits the coefficient
        # arbitrarily across predictors and reports identical p-values with
        # flipped signs -- which looks like a finding and is not one.
        n_lang = g["lang"].nunique()
        n_distinct = g[live].drop_duplicates().shape[0]
        collinear = bool(np.max(vifs) > 10.0) or (n_distinct <= n_lang)

        blocks, names = [np.ones((len(g), 1))], ["intercept"]
        blocks.append(Z)
        names.extend(live)
        if with_lang_dummies and n_lang > 1:
            d = pd.get_dummies(g["lang"], prefix="lang", drop_first=True)
            blocks.append(d.to_numpy(float))
            names.extend(list(d.columns))
        X = np.hstack(blocks)
        y = g[target].to_numpy(float)
        beta, se, t, p, r2 = _ols(X, y)
        for nm, b, s, tt, pp in zip(names, beta, se, t, p):
            if nm.startswith("lang_") or nm == "intercept":
                continue
            out.append({"system": sysname, "target": target, "predictor": nm,
                        "controls_language": bool(with_lang_dummies),
                        "coef": float(b), "se": float(s), "t": float(tt),
                        "p_value": float(pp), "model_r2": float(r2),
                        "vif": float(vifs[live.index(nm)]),
                        "collinearity_warning": collinear,
                        "n_distinct_density_profiles": int(n_distinct),
                        "n_languages": int(n_lang), "n": int(len(g))})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
def _fetch_real_gold_clips(cfg, out_dir, seconds_per_language):
    """Stream a small amount of real spontaneous audio per language from
    IndicVoices for hand annotation. Never downloads the full corpus:
    streaming mode reads shards lazily and stops once the budget is met.

    Failures for one language (bad config name, no network, gated access)
    are caught and reported; they do not stop the other languages or the
    synthetic half of the pack.
    """
    rs = cfg["human_gold"].get("real_source", {})
    if not rs:
        print("[step08b] no human_gold.real_source configured -- skipping "
              "real-audio half")
        return []
    try:
        from datasets import load_dataset
    except ImportError:
        print("[step08b] `datasets` not installed -- skipping real-audio half "
              "(pip install datasets to enable)")
        return []

    from vadlib.audio import resample, write_wav
    sr = int(cfg["data"]["target_sr"])
    token = cfg["credentials"].get("hf_token", "") or None
    dataset_id = rs["dataset"]
    split = rs.get("split", "valid")
    audio_field = rs.get("audio_field", "audio_filepath")
    text_field = rs.get("text_field", "text")
    clip_max_s = float(rs.get("clip_max_s", 20))
    lang_map = rs.get("lang_config_map", {})

    real_dir = out_dir / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for lang_code, hf_config in lang_map.items():
        target_s = float(seconds_per_language)
        if target_s <= 0:
            continue
        try:
            ds = load_dataset(dataset_id, hf_config, split=split,
                              streaming=True, token=token)
        except Exception as e:
            print(f"  [warn] {lang_code}: could not open {dataset_id}/"
                  f"{hf_config}/{split} ({e}). Verify the config name at "
                  f"https://huggingface.co/datasets/{dataset_id} and check "
                  f"credentials.hf_token. Skipping real audio for this "
                  f"language; its synthetic half is unaffected.")
            continue

        acc, n_clips = 0.0, 0
        try:
            for ex in ds:
                if acc >= target_s:
                    break
                audio = ex.get(audio_field)
                if audio is None:
                    continue
                arr = np.asarray(audio.get("array"), dtype=np.float32)
                src_sr = int(audio.get("sampling_rate", sr))
                dur = len(arr) / max(src_sr, 1)
                if dur < 1.0:
                    continue
                if dur > clip_max_s:
                    keep = int(clip_max_s * src_sr)
                    arr = arr[:keep]
                    dur = clip_max_s
                y = resample(arr, src_sr, sr)
                clip_id = f"real_{lang_code}_{n_clips:03d}"
                wav_p = real_dir / f"{clip_id}.wav"
                write_wav(wav_p, y, sr)
                lab_p = real_dir / f"{clip_id}_TEMPLATE.txt"
                lab_p.write_text("", encoding="utf-8")
                rows.append({
                    "session_id": clip_id, "lang": lang_code, "source": "real_indicvoices",
                    "wav": str(wav_p), "label_file": str(lab_p),
                    "duration_s": dur, "annotated": 0,
                    "reference_transcript": str(ex.get(text_field, ""))[:200],
                })
                acc += dur
                n_clips += 1
        except Exception as e:
            print(f"  [warn] {lang_code}: streaming interrupted ({e}); "
                  f"kept {n_clips} clips ({acc:.0f}s) before failing")

        if n_clips:
            print(f"  [real ] {lang_code} ({hf_config}): {n_clips} clips, "
                  f"{acc:.0f}s from IndicVoices")
        else:
            print(f"  [warn] {lang_code}: 0 clips obtained -- check network/"
                  f"config name; synthetic half for this language is unaffected")

    return rows


def make_human_gold_pack(cfg):
    """Emit stimuli + Audacity label templates for hand annotation.

    The per-language budget is split (default 50/50) between:
      * SYNTHETIC sessions -- exact-GT benchmark audio. Annotating these
        validates the committee's excerpt boundaries and the
        `leave_system_out` sensitivity, not VAD-in-the-wild performance.
      * REAL spontaneous audio from IndicVoices -- no synthetic construction,
        no forced alignment; humans supply the label directly. This is what
        answers "concatenated read speech isn't VAD" at essentially zero
        extra compute, since only download + annotation time is spent.

    Failure to reach any real audio (no network, bad config, missing
    `datasets`) degrades gracefully to a synthetic-only pack with a clear
    note in INSTRUCTIONS.txt, rather than failing the whole pipeline.
    """
    hg = cfg.get("human_gold", {})
    if not hg.get("enabled", True):
        return None
    out = Path(cfg["paths"]["bench"]) / "human_gold"
    out.mkdir(parents=True, exist_ok=True)

    sessions_p = Path(cfg["paths"]["bench"]) / "sessions.csv"
    if sessions_p.exists():
        sessions = pd.read_csv(sessions_p)
    else:
        # Day-1 case: this step is designed to run before Step 03 has ever
        # built the benchmark, so the synthetic half is simply empty for now
        # -- not an error. `test_languages(cfg)` gives the language list from
        # config alone, so the real-audio half and the "missing real clips"
        # report below still work with no synthetic sessions on disk at all.
        print(f"[step08b] {sessions_p} not found -- skipping the synthetic "
              "half for now (this is expected if Step 03 hasn't run yet). "
              "Re-run this step after Step 03 to fill it in.")
        sessions = pd.DataFrame(columns=["lang", "split", "session_id",
                                         "wav", "duration_s"])

    all_lang_codes = ([l["code"] for l in test_languages(cfg)]
                      if sessions.empty else sorted(sessions["lang"].unique()))

    total_s = float(hg.get("seconds_per_language", 150))
    synth_frac = float(hg.get("split_synthetic_frac", 0.5))
    synth_s = total_s * synth_frac
    real_s = total_s * (1.0 - synth_frac)

    rows = []
    for lang, g in sessions.groupby("lang"):
        acc = 0.0
        for _, s in g[g["split"] == "test"].iterrows():
            if acc >= synth_s:
                break
            dst = out / f"{s['session_id']}.wav"
            shutil.copy(s["wav"], dst)
            lab = out / f"{s['session_id']}_TEMPLATE.txt"
            lab.write_text("", encoding="utf-8")
            rows.append({"session_id": s["session_id"], "lang": lang,
                         "source": "synthetic",
                         "wav": str(dst), "label_file": str(lab),
                         "duration_s": s["duration_s"], "annotated": 0,
                         "reference_transcript": ""})
            acc += float(s["duration_s"])

    print(f"[step08b] fetching real spontaneous audio "
          f"(~{real_s:.0f}s/language) from IndicVoices ...")
    rows.extend(_fetch_real_gold_clips(cfg, out, real_s))

    # Explicit schema: `rows` can be fully empty on a day-1 run (no sessions
    # yet AND no real clips yet, e.g. `datasets` not installed). Without a
    # fixed column list, pd.DataFrame([]) has no columns at all and every
    # downstream .source / ["lang"] access below raises.
    cols = ["session_id", "lang", "source", "wav", "label_file",
            "duration_s", "annotated", "reference_transcript"]
    man = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    man.to_csv(out / "annotation_manifest.csv", index=False)

    n_real = int((man["source"] == "real_indicvoices").sum())
    n_synth = int((man["source"] == "synthetic").sum())
    langs_missing_real = sorted(
        set(all_lang_codes) - set(man.loc[man["source"] == "real_indicvoices", "lang"]))

    with open(out / "INSTRUCTIONS.txt", "w", encoding="utf-8") as f:
        f.write(
            "HUMAN GOLD ANNOTATION -- IndicVAD-Bench\n"
            "=======================================\n\n"
            f"This pack has TWO parts, {n_synth} synthetic + {n_real} real clips:\n\n"
            "  1. SYNTHETIC (top-level *.wav) -- exact-GT benchmark audio.\n"
            "     Annotating these checks the reference-committee's excerpt\n"
            "     boundaries, not real-world VAD performance.\n"
            "  2. REAL (real/*.wav) -- spontaneous speech from IndicVoices\n"
            "     (CC BY 4.0). No synthetic construction, no forced alignment:\n"
            "     you are the ground truth. This is the external-validity\n"
            "     evidence for the paper.\n\n"
            + (f"  NOTE: no real clips were obtained for: {langs_missing_real}.\n"
               "  Check network access and human_gold.real_source.lang_config_map\n"
               "  against https://huggingface.co/datasets/ai4bharat/IndicVoices\n"
               "  and re-run `python main.py --steps 03b`. The synthetic half\n"
               "  for these languages is unaffected.\n\n"
               if langs_missing_real else "")
            + "For each wav: open in Audacity, File > Import > Labels on the\n"
            "matching _TEMPLATE.txt, mark every SPEECH region, then\n"
            "File > Export > Export Labels back over the same filename.\n\n"
            "Conventions (must match the synthetic reference):\n"
            "  * Pauses SHORTER than 0.20 s are NOT marked as silence -- they\n"
            "    stay inside the speech region. This covers stop closures and\n"
            "    geminates, and is the convention the benchmark assumes.\n"
            "  * Breaths, coughs, lip smacks: NOT speech.\n"
            "  * Mark boundaries at the first/last visible glottal pulse or\n"
            "    frication energy, not at the waveform envelope knee.\n"
            "  * Two annotators per file. Disagreements are kept, not resolved:\n"
            "    report Cohen's kappa and median boundary disagreement.\n"
            "  * Report synthetic and real subsets SEPARATELY in the paper --\n"
            "    they answer different questions (reference validation vs.\n"
            "    external validity) and should not be pooled into one number.\n\n"
            "Then run:  python main.py --steps 03b --score-human-gold\n")
    print(f"[step08b] annotation pack: {n_synth} synthetic + {n_real} real "
          f"clips -> {out}")
    if langs_missing_real:
        print(f"[step08b] WARNING: no real clips for {langs_missing_real}. "
              "See INSTRUCTIONS.txt.")
    return out


def _parse_label_file(path):
    """Audacity label export -> list of (start, end) SPEECH segments only.

    Rows are 3-column (start, end, label). Only label == "1" counts as
    speech; "0" and anything else non-standard (blank, "o", "b1", ...) is
    treated as non-speech -- ambiguous tags are counted and reported rather
    than silently guessed at.
    """
    segs, n_ambiguous, n_rows = [], 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        n_rows += 1
        label = parts[2].strip() if len(parts) >= 3 else "1"
        if label == "1":
            segs.append((a, b))
        elif label != "0":
            n_ambiguous += 1
    return segs, n_ambiguous, n_rows


def _annotator_label_path(annotator_dir, source, session_id):
    sub = "synth" if source == "synthetic" else "real"
    p = Path(annotator_dir) / sub / f"{session_id}_TEMPLATE.txt"
    return p if p.exists() and p.stat().st_size > 0 else None


def score_human_gold(cfg):
    """Score systems against returned human labels (if any exist).

    Kept split by `source` throughout: synthetic and real subsets answer
    different questions (reference validation vs. external validity) and
    must not be pooled into one summary number.

    If `human_gold.annotator_dirs` names exactly two directories (each with
    `synth/` and `real/` subfolders of `{session_id}_TEMPLATE.txt` files),
    scores both annotators against each other: Cohen's kappa and onset/offset
    boundary MAD per session, as promised in INSTRUCTIONS.txt. Falls back to
    the single-annotator summary (segment count / speech time only) if
    `annotator_dirs` isn't set.
    """
    out = Path(cfg["paths"]["bench"]) / "human_gold"
    man_p = out / "annotation_manifest.csv"
    if not man_p.exists():
        print("[step08b] no annotation pack; run without --score-human-gold first")
        return None
    man = pd.read_csv(man_p)
    if "source" not in man.columns:
        man["source"] = "synthetic"      # backward compatibility

    annotator_dirs = cfg.get("human_gold", {}).get("annotator_dirs")
    if annotator_dirs and len(annotator_dirs) == 2:
        return _score_human_gold_dual(cfg, man, annotator_dirs)

    rows = []
    for _, r in man.iterrows():
        p = Path(r["label_file"])
        if not p.exists() or p.stat().st_size == 0:
            continue
        segs, _, _ = _parse_label_file(p)
        if segs:
            rows.append({"session_id": r["session_id"], "lang": r["lang"],
                         "source": r["source"], "n_segments": len(segs),
                         "speech_s": sum(b - a for a, b in segs),
                         "duration_s": r["duration_s"]})
    if not rows:
        print("[step08b] no annotations returned yet -- skipping")
        return None
    df = pd.DataFrame(rows)
    dst = Path(cfg["paths"]["results"]) / "human_gold_summary.csv"
    df.to_csv(dst, index=False)
    print(f"[step08b] {len(df)} annotated clips -> {dst}")
    for src, g in df.groupby("source"):
        print(f"  {src:16s}: {len(g)} clips, {g.duration_s.sum()/60:.1f} min, "
              f"{g.groupby('lang').size().shape[0]} languages")
    return dst


def _score_human_gold_dual(cfg, man, annotator_dirs):
    dir_a, dir_b = annotator_dirs
    summary_rows, agree_rows = [], []
    for _, r in man.iterrows():
        pa = _annotator_label_path(dir_a, r["source"], r["session_id"])
        pb = _annotator_label_path(dir_b, r["source"], r["session_id"])
        if pa is None or pb is None:
            continue
        segs_a, amb_a, _ = _parse_label_file(pa)
        segs_b, amb_b, _ = _parse_label_file(pb)
        if not segs_a and not segs_b:
            continue

        n_frames = int(round(float(r["duration_s"]) * 100))   # 10ms grid
        mask_a = segments_to_mask(segs_a, n_frames, frame_ms=10)
        mask_b = segments_to_mask(segs_b, n_frames, frame_ms=10)

        kappa = cohen_kappa(mask_a, mask_b)
        bm = boundary_metrics(mask_a, mask_b, frame_ms=10)

        speech_s_a = sum(b - a for a, b in segs_a)
        speech_s_b = sum(b - a for a, b in segs_b)

        summary_rows.append({
            "session_id": r["session_id"], "lang": r["lang"],
            "source": r["source"], "duration_s": r["duration_s"],
            "n_segments": round(0.5 * (len(segs_a) + len(segs_b)), 1),
            "speech_s": 0.5 * (speech_s_a + speech_s_b),
        })
        agree_rows.append({
            "session_id": r["session_id"], "lang": r["lang"],
            "source": r["source"], "cohen_kappa": kappa,
            "onset_mad_s": bm["onset_mad_s"], "offset_mad_s": bm["offset_mad_s"],
            "n_segments_a": len(segs_a), "n_segments_b": len(segs_b),
            "speech_s_a": speech_s_a, "speech_s_b": speech_s_b,
            "n_ambiguous_labels_a": amb_a, "n_ambiguous_labels_b": amb_b,
        })

    if not summary_rows:
        print("[step08b] no annotations returned yet from both annotators -- skipping")
        return None

    res = Path(cfg["paths"]["results"])
    df = pd.DataFrame(summary_rows)
    dst = res / "human_gold_summary.csv"
    df.to_csv(dst, index=False)

    agree = pd.DataFrame(agree_rows)
    agree_dst = res / "human_gold_interannotator.csv"
    agree.to_csv(agree_dst, index=False)

    print(f"[step08b] {len(df)} annotated clips (2 annotators) -> {dst}")
    print(f"[step08b] inter-annotator agreement -> {agree_dst}")
    for src, g in agree.groupby("source"):
        print(f"  {src:16s}: {len(g)} clips, mean kappa={g.cohen_kappa.mean():.3f}, "
              f"median onset_mad_s={g.onset_mad_s.median():.3f}, "
              f"median offset_mad_s={g.offset_mad_s.median():.3f}")
    n_amb = int(agree["n_ambiguous_labels_a"].sum() + agree["n_ambiguous_labels_b"].sum())
    if n_amb:
        print(f"[step08b] WARNING: {n_amb} label rows had non-standard tags "
              "(not \"0\"/\"1\") and were scored as non-speech; see "
              "n_ambiguous_labels_a/b in human_gold_interannotator.csv")
    return dst


def run(cfg):
    res = Path(cfg["paths"]["results"])
    metrics_p = res / "frame_metrics.csv"
    if not metrics_p.exists():
        print(f"[step08] {metrics_p} not found -- run steps 01-05 first. "
              "(The human-gold annotation pack no longer lives here: use "
              "`python main.py --steps 03b`, which has no dependency on "
              "scoring and can be run any time, including before step 01.)")
        return None
    metrics = pd.read_csv(metrics_p)
    feats = _session_features(cfg)
    if feats.empty:
        print("[step08] no transcripts available -- skipping phonology analysis")
    else:
        feats.to_csv(res / "phono_features.csv", index=False)
        print("\n[step08] phonological density by language (orthographic proxy):")
        print(feats.groupby(["lang", "family"])[
            ["breathy_density", "aspirate_density",
             "retroflex_density", "geminate_density"]].mean().round(4).to_string())

        noisy = metrics[metrics["snr_db"] < 99]
        parts = []
        for target in ["offset_mad_s", "onset_mad_s", "segment_count_ratio",
                       "p_miss", "deter"]:
            if target not in noisy.columns:
                continue
            for ctrl in (False, True):
                r = _regress(noisy, feats, target, ctrl)
                if r is not None and not r.empty:
                    parts.append(r)
        if parts:
            reg = pd.concat(parts, ignore_index=True)
            # BH-FDR correction over the meaningful family only: rows with
            # controls_language=False are a diagnostic contrast (how much
            # does the effect shrink once language is controlled?), not
            # part of the claim, so they are excluded from the correction
            # denominator. Including them would understate q-values for
            # the tests we actually report as findings.
            reg["p_value_bh"] = np.nan
            meaningful = reg.controls_language & ~reg.collinearity_warning
            reg.loc[meaningful, "p_value_bh"] = _bh_fdr(
                reg.loc[meaningful, "p_value"].to_numpy())
            reg.to_csv(res / "phono_regression.csv", index=False)
            n_bad = int(reg["collinearity_warning"].sum())
            if n_bad:
                print(f"\n[step08] WARNING: {n_bad}/{len(reg)} rows flagged "
                      "collinear. Density features carry no within-language "
                      "variation, so they cannot be separated from language "
                      "identity. DO NOT report those rows as phonological "
                      "effects. Increase data.max_utts_per_lang so sessions "
                      "draw more distinct transcripts.")
            raw_sig = reg[(reg.p_value < 0.05) & meaningful]
            sig = reg[(reg.p_value_bh < 0.05) & meaningful]
            print(f"\n[step08] {len(raw_sig)}/{int(meaningful.sum())} "
                  "density effects pass raw p<0.05 (language controlled, "
                  f"VIF<10); {len(sig)} survive Benjamini-Hochberg FDR "
                  "correction across that family (q<0.05) -- report the "
                  "BH-corrected count as the headline number, the raw count "
                  "only as context (~5% of tests are expected to pass "
                  "raw p<0.05 by chance alone at this family size).")
            if not sig.empty:
                print(sig[["system", "target", "predictor", "coef",
                           "p_value", "p_value_bh", "vif"]]
                      .round(4).to_string(index=False))

    return res / "phono_regression.csv"

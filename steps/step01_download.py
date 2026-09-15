"""Step 01 -- download source data.

FLEURS  (CC BY 4.0)  -> parallel FLoRes content across languages. This is the
                        methodological core: identical sentences in every
                        language means content is held constant.
OpenSLR-28           -> room impulse responses AND point-source noises in one
                        1.3 GB archive. Chosen over MUSAN (11 GB) deliberately.

COLAB STORAGE STRATEGY
----------------------
Archives (tarballs, zips) are written to DRIVE: a handful of large files that
must survive VM death, so you never re-download after a disconnect.
Extraction goes to LOCAL SCRATCH: thousands of small WAVs, which are
pathologically slow over Drive's FUSE mount.

Re-extraction after a session restart costs ~2 minutes. Re-downloading costs
30-90 minutes. That is the whole trade.

Idempotent: re-running skips completed downloads and completed extractions.
"""
from __future__ import annotations

import csv
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from vadlib.config import all_languages


def _download(url, dest, desc=""):
    """Stream to Drive with resume-safe .part naming and progress."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already on disk "
              f"({dest.stat().st_size/1e6:.0f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {desc or dest.name}")
    print(f"         {url}")
    with urllib.request.urlopen(url, timeout=180) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r         {done/1e6:8.0f} / {total/1e6:.0f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
        print()
    tmp.rename(dest)
    return dest


def _hf_download_to(repo, filename, dest, token=""):
    """Fetch one file from the HF hub and place it at `dest` (on Drive)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name}")
        return dest
    from huggingface_hub import hf_hub_download
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Cache into scratch, then copy the single artifact to Drive: this avoids
    # HF's symlinked blob tree ever touching the FUSE mount.
    src = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                          token=token or None)
    shutil.copy(src, dest)
    print(f"  [get ] {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
    return dest


def _parse_fleurs_tsv(path):
    """FLEURS TSVs are headerless with a variable column count.

    Parsed positionally-agnostically: the column ending in .wav is the file
    name, the longest text column is the transcription, and MALE/FEMALE marks
    gender. This survives schema drift across FLEURS releases.
    """
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for parts in csv.reader(f, delimiter="\t"):
            if not parts:
                continue
            fname = next((p for p in parts if p.strip().endswith(".wav")), None)
            if fname is None:
                continue
            texts = [p for p in parts if not p.strip().endswith(".wav")
                     and not p.strip().isdigit()]
            gender = next((p.lower() for p in parts
                           if p.strip().upper() in ("MALE", "FEMALE")), "unknown")
            rows.append({"file_name": fname.strip(),
                         "transcript": max(texts, key=len).strip() if texts else "",
                         "gender": gender})
    return rows


def download_fleurs(cfg):
    arch = Path(cfg["paths"]["archives"]) / "fleurs"
    raw = Path(cfg["paths"]["raw"]) / "fleurs"
    token = cfg["credentials"].get("hf_token", "")
    split = cfg["data"]["fleurs_split"]
    repo = cfg["data"]["fleurs_repo"]
    max_utts = int(cfg["data"]["max_utts_per_lang"])

    manifest = []
    for lang in all_languages(cfg):
        code = lang["code"]
        tar_dst = arch / code / f"{split}.tar.gz"
        tsv_dst = arch / code / f"{split}.tsv"
        out_dir = raw / code
        audio_dir = out_dir / "audio"

        # --- 1. archives -> DRIVE (download once, ever) ---------------------
        try:
            _hf_download_to(repo, f"data/{code}/{split}.tsv", tsv_dst, token)
            _hf_download_to(repo, f"data/{code}/audio/{split}.tar.gz",
                            tar_dst, token)
        except Exception as e:
            print(f"  [WARN] hub download failed for {code}: {e}")
            print("         falling back to datasets.load_dataset")
            _fallback_datasets(cfg, code, out_dir, split, max_utts)
            manifest.extend(_scan_dir(code, lang, out_dir, max_utts))
            continue

        # --- 2. extract -> LOCAL SCRATCH (fast; redone each session) --------
        done = out_dir / ".extracted"
        if done.exists() and not cfg["runtime"].get("overwrite", False):
            print(f"  [skip] {code} already extracted to scratch")
        else:
            audio_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_dst, "r:gz") as tf:
                n = 0
                for m in tf:
                    if not m.name.endswith(".wav"):
                        continue
                    if n >= max_utts * 2:
                        break
                    m.name = Path(m.name).name
                    tf.extract(m, audio_dir)
                    n += 1
            shutil.copy(tsv_dst, out_dir / "meta.tsv")
            done.touch()
            print(f"  [extr] {code}: {n} wavs -> scratch")

        rows = (_parse_fleurs_tsv(out_dir / "meta.tsv")
                if (out_dir / "meta.tsv").exists() else [])
        if rows:
            kept = 0
            for r in rows:
                wav = audio_dir / r["file_name"]
                if not wav.exists() or kept >= max_utts:
                    continue
                manifest.append({
                    "lang": code, "name": lang["name"], "family": lang["family"],
                    "control": int(bool(lang.get("control", False))),
                    "utt_id": Path(r["file_name"]).stem, "path": str(wav),
                    "transcript": r["transcript"], "gender": r["gender"]})
                kept += 1
        else:
            manifest.extend(_scan_dir(code, lang, out_dir, max_utts))
    return manifest


def _scan_dir(code, lang, out_dir, max_utts):
    wavs = sorted((out_dir / "audio").glob("*.wav"))[:max_utts]
    return [{"lang": code, "name": lang["name"], "family": lang["family"],
             "control": int(bool(lang.get("control", False))),
             "utt_id": w.stem, "path": str(w), "transcript": "",
             "gender": "unknown"} for w in wavs]


def _fallback_datasets(cfg, code, out_dir, split, max_utts):
    """Last resort if the hub file layout changed."""
    from datasets import load_dataset
    from vadlib.audio import write_wav
    ds = load_dataset(cfg["data"]["fleurs_repo"], code, split=split,
                      trust_remote_code=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(ds):
        if i >= max_utts:
            break
        a = ex["audio"]
        name = f"{code}_{i:05d}.wav"
        write_wav(out_dir / "audio" / name, a["array"], a["sampling_rate"])
        rows.append([str(i), name, ex.get("raw_transcription", ""),
                     ex.get("transcription", ""), str(len(a["array"])),
                     ex.get("gender", "unknown")])
    with open(out_dir / "meta.tsv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(rows)
    (out_dir / ".extracted").touch()


def download_corruption(cfg):
    arch = Path(cfg["paths"]["archives"])
    raw = Path(cfg["paths"]["raw"])

    z = _download(cfg["data"]["rirs_url"], arch / "rirs_noises.zip",
                  "OpenSLR-28 RIRs + point-source noises (1.3 GB) -> Drive")
    marker = raw / "RIRS_NOISES" / ".extracted"
    if marker.exists() and not cfg["runtime"].get("overwrite", False):
        print("  [skip] RIRS_NOISES already extracted to scratch")
    else:
        print("  [extr] RIRS_NOISES -> scratch ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()

    if cfg["data"].get("use_musan", False):
        t = _download(cfg["data"]["musan_url"], arch / "musan.tar.gz",
                      "MUSAN (11 GB) -> Drive")
        m = raw / "musan" / ".extracted"
        if not m.exists():
            print("  [extr] MUSAN -> scratch ...")
            with tarfile.open(t, "r:gz") as tf:
                tf.extractall(raw)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.touch()


def run(cfg):
    import pandas as pd
    print(f"[step01] storage mode : {cfg['storage_mode']}")
    print(f"[step01] archives ->    {cfg['paths']['archives']}   (persistent)")
    print(f"[step01] extract  ->    {cfg['paths']['raw']}   (scratch)\n")

    print("[step01] FLEURS ...")
    manifest = download_fleurs(cfg)
    print("\n[step01] corruption sources ...")
    download_corruption(cfg)

    df = pd.DataFrame(manifest)
    if df.empty:
        raise SystemExit("[step01] no utterances obtained -- check network/token")
    out = Path(cfg["paths"]["raw"]) / "utterances.csv"
    df.to_csv(out, index=False)
    print(f"\n[step01] {len(df)} utterances across {df['lang'].nunique()} languages")
    print(df.groupby(["lang", "family"]).size().to_string())
    print(f"[step01] wrote {out}")
    return out

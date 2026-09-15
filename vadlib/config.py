"""Configuration loading, storage layout, and device resolution.

STORAGE MODEL (this is the important part on Colab)
---------------------------------------------------
Google Drive is mounted over FUSE. It is fine for a handful of large files and
pathological for tens of thousands of small ones. Step 02 alone writes ~20k
excerpt WAVs; putting those on Drive turns a 30-minute step into hours and can
trip Drive API quotas mid-run.

So paths are split by access pattern:

  archives            -> DRIVE    few large downloads; survive session death
  raw/pool/bench/post -> SCRATCH  many small files; local disk speed
  results/paper/logs  -> DRIVE    small, precious, must outlive the VM

mode: local  -> everything under project.root (unchanged behaviour)
mode: colab  -> the split above, using storage.drive_root / storage.scratch_root
mode: auto   -> detect Colab
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml


class Config(dict):
    """dict with attribute access and dotted lookup."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def get_path(self, dotted, default=None):
        node = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def resolve_device(cfg) -> str:
    """Return 'cuda' or 'cpu'. `runtime.device: auto` detects."""
    want = str(cfg.get("runtime", {}).get("device", "auto")).lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if want == "cuda" and not has_cuda:
        print("[config] runtime.device=cuda but no CUDA visible; falling back to CPU")
        return "cpu"
    return "cuda" if has_cuda else "cpu"


def _layout(cfg):
    st = cfg.get("storage", {}) or {}
    mode = str(st.get("mode", "auto")).lower()
    if mode == "auto":
        mode = "colab" if in_colab() else "local"

    if mode == "colab":
        drive = Path(st.get("drive_root",
                            "/content/gdrive/MyDrive/indicvad")).expanduser()
        scratch = Path(st.get("scratch_root",
                              "/content/indicvad_scratch")).expanduser()
        if not drive.parent.exists():
            print(f"[config] WARNING: {drive.parent} does not exist. "
                  "Did you mount Drive? See the Colab notebook, cell 1.")
        return mode, {
            "root": scratch,
            "archives": drive / "00_archives",   # DRIVE: big, downloaded once
            "raw": scratch / "01_raw",           # scratch: extracted audio
            "pool": scratch / "02_pool",         # scratch: ~20k small wavs
            "bench": scratch / "03_benchmark",   # scratch
            "post": scratch / "04_posteriors",   # scratch
            "results": drive / "05_results",     # DRIVE: small + precious
            "paper": drive / "06_paper_pack",    # DRIVE
            "logs": drive / "logs",              # DRIVE
            "drive_root": drive,
            "scratch_root": scratch,
        }

    root = Path(cfg["project"]["root"]).expanduser().resolve()
    return mode, {
        "root": root, "archives": root / "00_archives", "raw": root / "01_raw",
        "pool": root / "02_pool", "bench": root / "03_benchmark",
        "post": root / "04_posteriors", "results": root / "05_results",
        "paper": root / "06_paper_pack", "logs": root / "logs",
        "drive_root": root, "scratch_root": root,
    }


def load_config(path="config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[config] not found: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))

    # Environment overrides the file, so secrets need not be committed.
    cfg.setdefault("credentials", {})
    for env_key, key in (("HF_TOKEN", "hf_token"), ("NTFY_TOPIC", "ntfy_topic")):
        v = os.environ.get(env_key, "").strip()
        if v:
            cfg["credentials"][key] = v

    mode, paths = _layout(cfg)
    cfg["storage_mode"] = mode
    cfg["paths"] = Config(paths)
    for k, v in paths.items():
        if k == "drive_root" and mode == "colab":
            continue
        Path(v).mkdir(parents=True, exist_ok=True)
    return cfg


def disk_report(cfg):
    """Free space on both volumes -- Colab local disk is the usual bottleneck."""
    out = {}
    for name in ("scratch_root", "drive_root"):
        p = Path(cfg["paths"][name])
        try:
            u = shutil.disk_usage(p)
            out[name] = {"path": str(p), "free_gb": u.free / 1e9,
                         "total_gb": u.total / 1e9}
        except Exception:
            out[name] = {"path": str(p), "free_gb": float("nan"),
                         "total_gb": float("nan")}
    return out


def test_languages(cfg):
    """Languages that appear in the benchmark (control languages excluded)."""
    return [l for l in cfg["languages"] if not l.get("control", False)]


def control_languages(cfg):
    return [l for l in cfg["languages"] if l.get("control", False)]


def all_languages(cfg):
    return list(cfg["languages"])


def lang_meta(cfg):
    return {l["code"]: l for l in cfg["languages"]}

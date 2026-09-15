"""Deterministic corruption of clean sessions.

Corrupted audio is generated on the fly rather than written to disk: the full
grid would be tens of gigabytes, and regeneration is cheap numpy. Determinism
comes from seeding on (session_idx, cond_id) -- NOT on language -- so every
language sees the identical noise waveform and room in a given cell.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio import (apply_rir, load_resampled, make_babble, mix_at_snr,
                    tile_to_length)


@lru_cache(maxsize=1)
def _index_rirs(raw_root, sr):
    root = Path(raw_root) / "RIRS_NOISES"
    rirs, noises = [], {"pointsource": [], "isotropic": []}
    if not root.exists():
        return tuple(), {"pointsource": tuple(), "isotropic": tuple()}
    for p in sorted((root / "simulated_rirs").rglob("*.wav")):
        rirs.append(str(p))
    for p in sorted((root / "real_rirs_isotropic_noises").rglob("*.wav")):
        (noises["isotropic"] if "noise" in p.name.lower() else rirs).append(str(p))
    for p in sorted((root / "pointsource_noises").rglob("*.wav")):
        noises["pointsource"].append(str(p))
    return tuple(rirs), {k: tuple(v) for k, v in noises.items()}


@lru_cache(maxsize=512)
def _cached_audio(path, sr):
    return load_resampled(path, sr)


@lru_cache(maxsize=1)
def _babble_sources(bench_root):
    p = Path(bench_root) / "babble_sources.json"
    if not p.exists():
        return tuple()
    with open(p) as f:
        return tuple(json.load(f))


def cond_seed(master_seed, session_idx, cond_id):
    """Seed depends on session index and condition -- never on language."""
    return (int(master_seed) * 7919 + int(session_idx) * 104729
            + (hash(cond_id) % 100003)) % (2 ** 32)


def corrupt(clean, sr, speech_mask_samples, cond, session_idx, cfg):
    """Apply one condition cell to a clean session waveform."""
    if cond["noise"] == "none" and cond["reverb"] == "none":
        return clean.astype(np.float32)

    rng = np.random.default_rng(
        cond_seed(cfg["project"]["seed"], session_idx, cond["cond_id"]))
    rirs, noise_bank = _index_rirs(str(cfg["paths"]["raw"]), sr)

    x = clean.astype(np.float32)
    if cond["reverb"] == "rir" and rirs:
        rir = _cached_audio(rirs[int(rng.integers(0, len(rirs)))], sr)
        x = apply_rir(x, rir)

    ntype = cond["noise"]
    n = len(x)
    if ntype == "babble":
        srcs = _babble_sources(str(cfg["paths"]["bench"]))
        excerpts = [_cached_audio(s, sr) for s in
                    rng.choice(srcs, size=min(len(srcs), 24), replace=False)] \
            if srcs else []
        noise = make_babble(n, excerpts, int(cfg["conditions"]["babble_n_talkers"]), rng)
    elif ntype in noise_bank and noise_bank[ntype]:
        bank = noise_bank[ntype]
        src = _cached_audio(bank[int(rng.integers(0, len(bank)))], sr)
        noise = tile_to_length(n, src, rng)
    elif ntype == "none":
        return x
    else:
        noise = rng.standard_normal(n).astype(np.float32)

    return mix_at_snr(
        x, noise, float(cond["snr_db"]), speech_mask_samples,
        level_normalization=cfg["conditions"].get(
            "level_normalization", "match_speech_rms"),
        verify=bool(cfg["conditions"].get("verify_snr", True)))


def speech_mask_samples(intervals, n_samples, sr):
    m = np.zeros(n_samples, dtype=bool)
    for a, b in intervals:
        i0, i1 = int(a * sr), min(n_samples, int(b * sr))
        if i1 > i0:
            m[i0:i1] = True
    return m


def speech_mask_frames(intervals, n_frames, frame_ms=10):
    m = np.zeros(int(n_frames), dtype=bool)
    fr = frame_ms / 1000.0
    for a, b in intervals:
        i0, i1 = int(round(a / fr)), min(int(n_frames), int(round(b / fr)))
        if i1 > i0:
            m[i0:i1] = True
    return m

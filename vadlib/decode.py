"""Decoding: frame posteriors -> binary speech decisions -> segments.

Kept separate from the models on purpose. Step 07 sweeps these four scalars
over CACHED posteriors, which is what makes the hyperparameter experiment cost
seconds instead of hours -- and it is the mechanism by which apparent
"language dependence" may turn out to be pure post-processing.
"""
from __future__ import annotations

import numpy as np


def decode(posterior, frame_ms=10, threshold=0.5, min_speech_s=0.0,
           min_silence_s=0.0, pad_s=0.0, hysteresis=0.15):
    """Convert a frame posterior sequence to a binary mask.

    hysteresis: speech must exceed `threshold` to start but only
    `threshold - hysteresis` to continue (Schmitt trigger). Set to 0 to disable.
    """
    p = np.asarray(posterior, dtype=np.float32).ravel()
    if p.size == 0:
        return np.zeros(0, dtype=bool)

    hi = float(threshold)
    lo = max(0.0, hi - float(hysteresis))
    mask = np.zeros(p.size, dtype=bool)
    on = False
    for i, v in enumerate(p):
        if on:
            on = v >= lo
        else:
            on = v >= hi
        mask[i] = on

    fr = frame_ms / 1000.0
    mask = _drop_short(mask, True, int(round(min_silence_s / fr)))   # bridge gaps
    mask = _drop_short(mask, False, int(round(min_speech_s / fr)))   # kill blips
    if pad_s > 0:
        mask = _dilate(mask, int(round(pad_s / fr)))
    return mask


def _runs(mask):
    """Yield (start, end_exclusive, value) runs of a boolean array."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    change = np.flatnonzero(np.diff(m.astype(np.int8))) + 1
    bounds = np.concatenate(([0], change, [m.size]))
    return [(int(bounds[i]), int(bounds[i + 1]), bool(m[bounds[i]]))
            for i in range(len(bounds) - 1)]


def _drop_short(mask, fill_value, min_len):
    """Runs of (not fill_value) shorter than min_len become fill_value."""
    if min_len <= 1:
        return mask
    out = np.asarray(mask, dtype=bool).copy()
    for s, e, v in _runs(mask):
        if v != fill_value and (e - s) < min_len:
            out[s:e] = fill_value
    return out


def _dilate(mask, k):
    if k <= 0:
        return mask
    out = np.asarray(mask, dtype=bool).copy()
    for s, e, v in _runs(mask):
        if v:
            out[max(0, s - k):min(len(out), e + k)] = True
    return out


def mask_to_segments(mask, frame_ms=10):
    """Boolean frame mask -> list of (start_s, end_s) speech segments."""
    fr = frame_ms / 1000.0
    return [(s * fr, e * fr) for s, e, v in _runs(mask) if v]


def segments_to_mask(segments, n_frames, frame_ms=10):
    fr = frame_ms / 1000.0
    m = np.zeros(int(n_frames), dtype=bool)
    for s, e in segments:
        i0 = max(0, int(round(s / fr)))
        i1 = min(int(n_frames), int(round(e / fr)))
        if i1 > i0:
            m[i0:i1] = True
    return m


def resample_posterior(p, n_target):
    """Linearly resample a posterior sequence onto the 10 ms scoring grid."""
    p = np.asarray(p, dtype=np.float32).ravel()
    if p.size == 0:
        return np.zeros(n_target, dtype=np.float32)
    if p.size == n_target:
        return p
    xi = np.linspace(0.0, 1.0, p.size)
    xo = np.linspace(0.0, 1.0, int(n_target))
    return np.interp(xo, xi, p).astype(np.float32)

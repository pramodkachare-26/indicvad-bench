"""VAD evaluation metrics.

Detection:  P_miss, P_fa, DetER (DIHARD SAD convention), DCF (NIST OpenSAD),
            ROC-AUC, EER.
Boundary:   onset/offset absolute deviation, boundary F1 at tolerances,
            segment-count ratio (the geminate over-splitting probe).
"""
from __future__ import annotations

import numpy as np

from .decode import _runs, mask_to_segments

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Collar
# --------------------------------------------------------------------------- #
def collar_mask(ref, frame_ms=10, collar_s=0.0):
    """True for frames to SCORE (i.e. outside the forgiveness collar)."""
    keep = np.ones(len(ref), dtype=bool)
    if collar_s <= 0:
        return keep
    k = int(round(collar_s / (frame_ms / 1000.0)))
    for s, e, _ in _runs(ref):
        keep[max(0, s - k):min(len(keep), s + k)] = False
        keep[max(0, e - k):min(len(keep), e + k)] = False
    return keep


# --------------------------------------------------------------------------- #
# Detection metrics
# --------------------------------------------------------------------------- #
def detection_metrics(ref, hyp, frame_ms=10, collar_s=0.0,
                      w_miss=0.75, w_fa=0.25):
    ref = np.asarray(ref, dtype=bool)
    hyp = np.asarray(hyp, dtype=bool)[: len(ref)]
    if len(hyp) < len(ref):
        hyp = np.pad(hyp, (0, len(ref) - len(hyp)))

    keep = collar_mask(ref, frame_ms, collar_s)
    r, h = ref[keep], hyp[keep]
    fr = frame_ms / 1000.0

    n_sp = int(r.sum())
    n_ns = int((~r).sum())
    miss = int((r & ~h).sum())
    fa = int((~r & h).sum())

    p_miss = miss / max(n_sp, 1)
    p_fa = fa / max(n_ns, 1)
    deter = (miss + fa) * fr / max(n_sp * fr, EPS)   # DIHARD SAD: err / ref speech
    acc = float((r == h).mean()) if r.size else float("nan")

    return dict(
        n_speech_s=n_sp * fr, n_nonspeech_s=n_ns * fr,
        miss_s=miss * fr, fa_s=fa * fr,
        p_miss=p_miss, p_fa=p_fa,
        deter=deter, dcf=w_miss * p_miss + w_fa * p_fa,
        accuracy=acc, speech_prior=n_sp / max(n_sp + n_ns, 1),
    )


def roc_auc(ref, posterior):
    """Rank-based AUC. No sklearn dependency; handles ties correctly."""
    y = np.asarray(ref, dtype=bool).ravel()
    s = np.asarray(posterior, dtype=np.float64).ravel()[: len(y)]
    if len(s) < len(y):
        s = np.pad(s, (0, len(y) - len(s)))
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def eer(ref, posterior, n_points=256):
    y = np.asarray(ref, dtype=bool).ravel()
    s = np.asarray(posterior, dtype=np.float64).ravel()[: len(y)]
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan"), float("nan")
    ths = np.quantile(s, np.linspace(0.001, 0.999, n_points))
    best, best_t, eer_v = 1.0, 0.5, 1.0
    for t in ths:
        h = s >= t
        pm = float((y & ~h).sum()) / max(int(y.sum()), 1)
        pf = float((~y & h).sum()) / max(int((~y).sum()), 1)
        if abs(pm - pf) < best:
            best, best_t = abs(pm - pf), float(t)
            eer_v = 0.5 * (pm + pf)
    return float(eer_v), best_t


# --------------------------------------------------------------------------- #
# Boundary metrics
# --------------------------------------------------------------------------- #
def _boundaries(mask, frame_ms=10):
    segs = mask_to_segments(mask, frame_ms)
    return np.array([s for s, _ in segs]), np.array([e for _, e in segs])


def boundary_metrics(ref, hyp, frame_ms=10, tolerances=(0.02, 0.05, 0.10, 0.20),
                     match_window_s=1.0):
    r_on, r_off = _boundaries(ref, frame_ms)
    h_on, h_off = _boundaries(hyp, frame_ms)

    out = {
        "n_ref_segments": int(len(r_on)),
        "n_hyp_segments": int(len(h_on)),
        "segment_count_ratio": (len(h_on) / len(r_on)) if len(r_on) else float("nan"),
    }
    for name, rb, hb in (("onset", r_on, h_on), ("offset", r_off, h_off)):
        dev = _match_deviations(rb, hb, match_window_s)
        out[f"{name}_mad_s"] = float(np.median(np.abs(dev))) if dev.size else float("nan")
        out[f"{name}_iqr_s"] = float(np.subtract(*np.percentile(np.abs(dev), [75, 25]))) \
            if dev.size else float("nan")
        out[f"{name}_bias_s"] = float(np.median(dev)) if dev.size else float("nan")

    allr = np.concatenate([r_on, r_off]) if len(r_on) else np.array([])
    allh = np.concatenate([h_on, h_off]) if len(h_on) else np.array([])
    for tol in tolerances:
        p, rc, f1 = _boundary_prf(allr, allh, tol)
        key = f"{int(round(tol * 1000))}ms"
        out[f"bF1_{key}"] = f1
        out[f"bPrec_{key}"] = p
        out[f"bRec_{key}"] = rc
    return out


def _match_deviations(ref_b, hyp_b, window_s):
    """Greedy nearest-neighbour matching; returns hyp-minus-ref deviations."""
    if len(ref_b) == 0 or len(hyp_b) == 0:
        return np.array([])
    used = np.zeros(len(hyp_b), dtype=bool)
    devs = []
    for rb in ref_b:
        d = np.abs(hyp_b - rb)
        d[used] = np.inf
        j = int(np.argmin(d))
        if np.isfinite(d[j]) and d[j] <= window_s:
            used[j] = True
            devs.append(float(hyp_b[j] - rb))
    return np.array(devs)


def _boundary_prf(ref_b, hyp_b, tol):
    if len(ref_b) == 0 or len(hyp_b) == 0:
        return float("nan"), float("nan"), float("nan")
    used = np.zeros(len(hyp_b), dtype=bool)
    tp = 0
    for rb in ref_b:
        d = np.abs(hyp_b - rb)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
    prec = tp / len(hyp_b)
    rec = tp / len(ref_b)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def score_all(ref, posterior, hyp, frame_ms=10, collars=(0.0, 0.25),
              tolerances=(0.02, 0.05, 0.10, 0.20), match_window_s=1.0,
              w_miss=0.75, w_fa=0.25):
    """One row of results for a (system, session, condition) triple."""
    row = {}
    for c in collars:
        d = detection_metrics(ref, hyp, frame_ms, c, w_miss, w_fa)
        suffix = "" if c == 0.0 else f"_c{int(round(c * 1000))}"
        for k, v in d.items():
            row[k + suffix] = v
    row["auc"] = roc_auc(ref, posterior)
    e, t = eer(ref, posterior)
    row["eer"], row["eer_threshold"] = e, t
    row.update(boundary_metrics(ref, hyp, frame_ms, tolerances, match_window_s))
    return row

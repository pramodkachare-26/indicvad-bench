"""Inter-rater agreement for the reference-construction committee.

The benchmark's reference labels come from unanimous agreement among several
independent detectors, not from a standard. That is defensible -- but only if
the committee is shown to be stable rather than arbitrary. A reviewer will ask
"did the three systems actually agree, or did unanimity just never trigger?"

These functions answer that with the conventional statistics:

  Fleiss' kappa  -- multi-rater agreement above chance (the headline number)
  Cohen's kappa  -- pairwise, to expose an outlier committee member
  observed / expected agreement -- the raw inputs, so the kappa is auditable

Interpretation (Landis & Koch): >0.80 almost perfect, 0.61-0.80 substantial,
0.41-0.60 moderate. Below ~0.6 the reference is shaky and the unanimity rule
is doing more work than the systems are.
"""
from __future__ import annotations

import numpy as np


def observed_agreement(votes):
    """Fraction of frames on which every rater agrees. votes: (n_raters, T)."""
    v = np.asarray(votes, dtype=bool)
    if v.size == 0:
        return float("nan")
    return float((v.all(axis=0) | (~v).all(axis=0)).mean())


def fleiss_kappa(votes):
    """Fleiss' kappa for n binary raters over T items.

    votes : (n_raters, T) boolean array.
    """
    v = np.asarray(votes, dtype=bool)
    if v.ndim != 2 or v.shape[0] < 2 or v.shape[1] == 0:
        return float("nan")
    n_raters, T = v.shape

    n_speech = v.sum(axis=0).astype(np.float64)          # per frame
    n_sil = n_raters - n_speech

    # Per-item agreement P_i
    P_i = (n_speech ** 2 + n_sil ** 2 - n_raters) / (n_raters * (n_raters - 1))
    P_bar = float(P_i.mean())

    # Chance agreement from marginal category proportions
    p_speech = float(n_speech.sum() / (T * n_raters))
    p_sil = 1.0 - p_speech
    P_e = p_speech ** 2 + p_sil ** 2

    if abs(1.0 - P_e) < 1e-12:
        # Degenerate: every rater called everything the same class.
        return float("nan")
    return float((P_bar - P_e) / (1.0 - P_e))


def cohen_kappa(a, b):
    """Cohen's kappa between two binary raters."""
    a = np.asarray(a, dtype=bool).ravel()
    b = np.asarray(b, dtype=bool).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa1, pb1 = float(a.mean()), float(b.mean())
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def committee_report(votes, names):
    """Full agreement summary for one committee over pooled frames."""
    v = np.asarray(votes, dtype=bool)
    rep = {
        "n_raters": int(v.shape[0]),
        "n_frames": int(v.shape[1]) if v.ndim == 2 else 0,
        "fleiss_kappa": fleiss_kappa(v),
        "observed_unanimity": observed_agreement(v),
        "unanimous_speech_frac": float(v.all(axis=0).mean()) if v.size else float("nan"),
    }
    for i in range(len(names)):
        rep[f"speech_rate_{names[i]}"] = float(v[i].mean())
        for j in range(i + 1, len(names)):
            rep[f"cohen_kappa_{names[i]}_vs_{names[j]}"] = cohen_kappa(v[i], v[j])
    return rep


def interpret(k):
    if not np.isfinite(k):
        return "undefined"
    if k > 0.80:
        return "almost perfect"
    if k > 0.60:
        return "substantial"
    if k > 0.40:
        return "moderate"
    if k > 0.20:
        return "fair"
    return "slight"

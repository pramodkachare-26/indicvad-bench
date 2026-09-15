#!/usr/bin/env python3
"""Self-test: exercises the numpy/scipy core with no network and no torch.

Run this FIRST on the target machine:  python selftest.py
It validates I/O, corruption, decoding, metrics, the variance decomposition
and the phonological feature extractor against known-answer cases.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from vadlib.audio import (active_rms, apply_rir, mix_at_snr, read_wav, rms,
                          write_wav, load_resampled, make_babble)
from vadlib.decode import decode, mask_to_segments, segments_to_mask
from vadlib.metrics import detection_metrics, roc_auc, boundary_metrics
from vadlib.systems import EnergyVAD, LTSDVAD

OK, FAIL = 0, 0


def check(name, cond, extra=""):
    global OK, FAIL
    if cond:
        OK += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {extra}")


def synth_session(sr=16000, dur=20.0, seed=0):
    """Speech-like bursts (harmonic + noise) separated by true silence."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    x = np.zeros(n, dtype=np.float32)
    ivs, t = [], 1.0
    while t < dur - 2.0:
        seg = rng.uniform(0.8, 2.0)
        a, b = int(t * sr), int((t + seg) * sr)
        tt = np.arange(b - a) / sr
        f0 = rng.uniform(100, 200)
        sig = sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in range(1, 12))
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.5 * tt)      # syllabic modulation
        x[a:b] = (sig * env * 0.25).astype(np.float32)
        ivs.append((t, t + seg))
        t += seg + rng.uniform(0.4, 1.2)
    return x, ivs


def main():
    sr = 16000
    print("\n[1] audio I/O round-trip")
    with tempfile.TemporaryDirectory() as td:
        x = (np.random.default_rng(0).standard_normal(sr) * 0.1).astype(np.float32)
        p = Path(td) / "t.wav"
        write_wav(p, x, sr)
        y, sr2 = read_wav(p)
        check("sample rate preserved", sr2 == sr)
        check("waveform round-trips", np.max(np.abs(x - y)) < 2e-4,
              f"max err {np.max(np.abs(x-y)):.2e}")
        check("resample 16k->8k halves length",
              abs(len(load_resampled(p, 8000)) - sr // 2) <= 2)

    print("\n[2] active-speech SNR mixing")
    x, ivs = synth_session(sr)
    mask = np.zeros(len(x), dtype=bool)
    for a, b in ivs:
        mask[int(a * sr):int(b * sr)] = True
    noise = np.random.default_rng(1).standard_normal(len(x)).astype(np.float32)
    s_rms = active_rms(x, mask)

    # Raw additive mixing: output-minus-input IS the scaled noise.
    for target in (20, 10, 0, -5):
        y = mix_at_snr(x, noise, target, mask, level_normalization="none",
                       verify=False)
        got = 20 * np.log10(s_rms / max(rms(y - x), 1e-12))
        check(f"SNR {target:+3d} dB achieved (no level norm)",
              abs(got - target) < 0.5, f"got {got:.2f}")

    # With level normalization the output is g*(speech + noise). Recover the
    # components explicitly -- (y - x) is NOT the noise once a gain is applied,
    # which is precisely the mistake the mix_at_snr docstring warns about.
    print("\n[2b] level normalization preserves SNR exactly")
    for target in (20, 10, 0, -5):
        y = mix_at_snr(x, noise, target, mask,
                       level_normalization="match_speech_rms", verify=False)
        tgt_n = s_rms / (10.0 ** (target / 20.0))
        n_scaled = noise * (tgt_n / rms(noise))
        g = s_rms / rms(x + n_scaled)
        got = 20 * np.log10(active_rms(g * x, mask) / max(rms(g * n_scaled), 1e-12))
        check(f"SNR {target:+3d} dB preserved under global gain",
              abs(got - target) < 0.01, f"got {got:.3f}")
        check(f"output RMS == speech RMS at {target:+3d} dB",
              abs(rms(y) / s_rms - 1.0) < 0.01, f"ratio {rms(y)/s_rms:.3f}")

    # The confound this fixes: without normalization, level tracks SNR.
    lv = {t: rms(mix_at_snr(x, noise, t, mask, level_normalization="none",
                            verify=False)) / s_rms for t in (20, -5)}
    check("uncorrected mixing does inflate level (the confound)",
          lv[-5] / lv[20] > 1.5, f"{lv}")

    print("\n[3] RIR convolution preserves level and length")
    rir = np.zeros(2000, dtype=np.float32)
    rir[100] = 1.0
    rir[300:800] = np.random.default_rng(2).standard_normal(500) * 0.05
    y = apply_rir(x, rir)
    check("length preserved", len(y) == len(x))
    check("level preserved", abs(rms(y) - rms(x)) / rms(x) < 0.05)
    # use a broadband probe: the sparse synth session gives an unstable argmax
    probe = np.random.default_rng(9).standard_normal(sr * 2).astype(np.float32)
    yp = apply_rir(probe, rir)
    lag = int(np.argmax(np.abs(np.correlate(yp, probe, "full"))) - (len(probe) - 1))
    check("direct path delay compensated", lag == 0, f"lag={lag}")

    print("\n[4] decoding: duration constraints behave")
    p = np.zeros(1000, dtype=np.float32)
    p[100:110] = 0.9          # 100 ms blip
    p[300:600] = 0.9          # 3 s speech
    p[420:430] = 0.0          # 100 ms internal dropout
    m0 = decode(p, 10, 0.5, 0.0, 0.0, 0.0, hysteresis=0.0)
    check("raw decode finds both regions", len(mask_to_segments(m0, 10)) == 3)
    m1 = decode(p, 10, 0.5, 0.20, 0.0, 0.0, hysteresis=0.0)
    check("min_speech=200ms removes the blip",
          all((b - a) >= 0.19 for a, b in mask_to_segments(m1, 10)))
    m2 = decode(p, 10, 0.5, 0.0, 0.20, 0.0, hysteresis=0.0)
    check("min_silence=200ms bridges the dropout",
          len(mask_to_segments(m2, 10)) == 2)
    m3 = decode(p, 10, 0.5, 0.0, 0.0, 0.10, hysteresis=0.0)
    check("pad extends segments", m3.sum() > m0.sum())

    print("\n[5] metrics: known-answer cases")
    ref = np.zeros(1000, dtype=bool); ref[200:700] = True
    d = detection_metrics(ref, ref.copy(), 10, 0.0)
    check("perfect hyp -> DetER 0", abs(d["deter"]) < 1e-9)
    d = detection_metrics(ref, np.zeros(1000, dtype=bool), 10, 0.0)
    check("empty hyp -> p_miss 1", abs(d["p_miss"] - 1.0) < 1e-9)
    check("empty hyp -> p_fa 0", abs(d["p_fa"]) < 1e-9)
    d = detection_metrics(ref, np.ones(1000, dtype=bool), 10, 0.0)
    check("all-speech hyp -> p_fa 1", abs(d["p_fa"] - 1.0) < 1e-9)
    check("all-speech DetER = nonspeech/speech ratio",
          abs(d["deter"] - 500 / 500) < 1e-6, f"{d['deter']}")
    shifted = np.zeros(1000, dtype=bool); shifted[210:710] = True
    d0 = detection_metrics(ref, shifted, 10, 0.0)
    dc = detection_metrics(ref, shifted, 10, 0.25)
    check("collar reduces measured error", dc["deter"] < d0["deter"])

    check("AUC perfect separation = 1.0",
          abs(roc_auc(ref, ref.astype(float)) - 1.0) < 1e-9)
    check("AUC constant score = 0.5",
          abs(roc_auc(ref, np.full(1000, 0.5)) - 0.5) < 1e-9)
    rng = np.random.default_rng(3)
    check("AUC random ~ 0.5", abs(roc_auc(ref, rng.random(1000)) - 0.5) < 0.06)

    b = boundary_metrics(ref, shifted, 10)
    check("onset deviation recovered as +0.10 s",
          abs(b["onset_mad_s"] - 0.10) < 1e-6, f"{b['onset_mad_s']}")
    check("segment count ratio = 1", abs(b["segment_count_ratio"] - 1.0) < 1e-9)

    split = np.zeros(1000, dtype=bool)
    split[200:400] = True; split[450:700] = True
    b2 = boundary_metrics(ref, split, 10)
    check("over-splitting shows ratio 2.0",
          abs(b2["segment_count_ratio"] - 2.0) < 1e-9)

    print("\n[6] signal-processing VADs separate speech from silence")
    for cls in (EnergyVAD, LTSDVAD):
        v = cls().load()
        clean, ivs = synth_session(sr, 20.0, seed=5)
        n_fr = len(v.posterior(clean, sr))
        ref = segments_to_mask(ivs, n_fr, 10)
        auc = roc_auc(ref, v.posterior(clean, sr))
        check(f"{cls.__name__} AUC > 0.85 on clean", auc > 0.85, f"AUC={auc:.3f}")

    print("\n[7] babble synthesis")
    ex = [synth_session(sr, 8.0, seed=i)[0] for i in range(6)]
    bab = make_babble(sr * 5, ex, 4, np.random.default_rng(0))
    check("babble has requested length", len(bab) == sr * 5)
    check("babble is unit-RMS normalised", abs(rms(bab) - 1.0) < 0.05)
    silent = [np.zeros(sr * 4, dtype=np.float32) for _ in range(4)]
    fb = make_babble(sr * 5, silent, 4, np.random.default_rng(0))
    check("degenerate excerpts fall back to noise, not silence", rms(fb) > 1e-3)

    print("\n[8] variance decomposition recovers a planted effect")
    from steps.step06_stats import variance_components
    import pandas as pd
    rng = np.random.default_rng(7)
    n = 1200
    df = pd.DataFrame({
        "lang": rng.choice(list("ABCDEFGH"), n),
        "snr_db": rng.choice([20, 15, 10, 5, 0, -5], n),
        "noise": rng.choice(["a", "b", "c"], n),
        "reverb": rng.choice(["none", "rir"], n),
    })
    # SNR dominates by construction; language contributes almost nothing.
    df["y"] = (-0.12 * df.snr_db
               + 0.02 * df.lang.map({c: i for i, c in enumerate("ABCDEFGH")})
               + 0.30 * (df.reverb == "rir")
               + rng.normal(0, 0.25, n))
    vc = variance_components(df, ["lang", "snr_db", "noise", "reverb"], "y")
    w = dict(zip(vc.factor, vc.omega2))
    check("SNR identified as dominant factor",
          w["snr_db"] > max(w["lang"], w["noise"], w["reverb"]),
          str({k: round(v, 3) for k, v in w.items()}))
    check("language share correctly small", w["lang"] < 0.05, f"{w['lang']:.4f}")

    print("\n[8b] committee agreement statistics")
    from vadlib.agreement import cohen_kappa, committee_report, fleiss_kappa
    rng = np.random.default_rng(11)
    T = 20000
    truth = rng.random(T) < 0.55

    def rater(flip):
        v = truth.copy()
        idx = rng.random(T) < flip
        v[idx] = ~v[idx]
        return v

    check("identical raters -> kappa 1.0",
          abs(fleiss_kappa(np.stack([truth] * 3)) - 1.0) < 1e-9)
    check("degenerate single-class -> kappa undefined",
          not np.isfinite(fleiss_kappa(np.ones((3, 100), dtype=bool))))
    k_clean = fleiss_kappa(np.stack([rater(0.01) for _ in range(3)]))
    k_noisy = fleiss_kappa(np.stack([rater(0.30) for _ in range(3)]))
    check("kappa decreases with rater disagreement", k_clean > k_noisy,
          f"{k_clean:.3f} vs {k_noisy:.3f}")
    check("clean committee reads as almost perfect", k_clean > 0.90,
          f"{k_clean:.3f}")
    V = np.stack([rater(0.05) for _ in range(3)])
    check("Fleiss ~ Cohen for symmetric raters",
          abs(fleiss_kappa(V) - cohen_kappa(V[0], V[1])) < 0.05)
    rep = committee_report(V, ["a", "b", "c"])
    check("committee_report emits pairwise kappas",
          sum(1 for k in rep if k.startswith("cohen_kappa_")) == 3, str(list(rep)))

    print("\n[9] phonological density extractor")
    from steps.step08_phonology import phono_features
    f_hi = phono_features("भारत में घर धीरे धीरे बदल रहा है")   # breathy-rich Hindi
    f_ta = phono_features("இந்தியாவில் வீடு மெதுவாக மாறுகிறது")  # Tamil, no breathy series
    check("Hindi breathy density > 0", f_hi["breathy_density"] > 0,
          str(f_hi))
    check("Tamil breathy density == 0", f_ta["breathy_density"] == 0.0,
          str(f_ta))
    f_gem = phono_features("पत्ता कच्चा")
    check("geminate detected", f_gem["geminate_density"] > 0, str(f_gem))
    check("non-Indic text returns None", phono_features("hello world") is None)

    print("\n" + "=" * 60)
    print(f"  {OK} passed, {FAIL} failed")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

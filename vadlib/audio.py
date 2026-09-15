"""Audio primitives: I/O, resampling, active-speech SNR mixing, reverberation.

Deliberately depends only on numpy + scipy so the corruption pipeline is
fully testable without torch/librosa/soundfile.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, resample_poly

EPS = 1e-10


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read_wav(path):
    """Read a PCM wav -> (float32 mono in [-1,1], sr). Falls back to soundfile."""
    path = str(path)
    try:
        with wave.open(path, "rb") as w:
            n_ch, width, sr, n_frames = (
                w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            )
            raw = w.readframes(n_frames)
        if width == 2:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif width == 1:
            x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported sample width {width}")
        if n_ch > 1:
            x = x.reshape(-1, n_ch).mean(axis=1)
        return np.ascontiguousarray(x, dtype=np.float32), sr
    except Exception:
        import soundfile as sf  # optional fallback for flac/mp3/24-bit
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return np.ascontiguousarray(x, dtype=np.float32), sr


def write_wav(path, x, sr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    g = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def load_resampled(path, sr_out):
    x, sr = read_wav(path)
    return resample(x, sr, sr_out)


# --------------------------------------------------------------------------- #
# Level / mixing
# --------------------------------------------------------------------------- #
def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + EPS))


def active_rms(x, speech_mask_samples):
    """RMS over reference-speech samples only (active-speech level)."""
    m = np.asarray(speech_mask_samples, dtype=bool)
    if m.sum() == 0:
        return rms(x)
    return rms(np.asarray(x)[m])


def tile_to_length(n, src, rng):
    """Return `n` samples from `src`, looping or random-cropping as needed."""
    src = np.asarray(src, dtype=np.float32)
    if len(src) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(src) >= n:
        off = int(rng.integers(0, len(src) - n + 1))
        return src[off:off + n].copy()
    reps = int(np.ceil(n / len(src)))
    return np.tile(src, reps)[:n].copy()


def mix_at_snr(speech, noise, snr_db, speech_mask_samples,
               level_normalization="match_speech_rms", verify=True,
               tolerance_db=0.25):
    """Scale `noise` so active-speech SNR equals `snr_db`, then add and re-level.

    SNR definition
    --------------
    SNR_dB = 20 log10( RMS(speech over reference speech frames)
                       / RMS(scaled noise over the whole signal) )

    Speech level is measured over reference speech regions ONLY. This matters:
    sessions are majority silence by construction (inserted pauses up to 2.5 s),
    so whole-signal RMS would understate speech level and silently shift the
    true SNR downward.

    Level normalization
    -------------------
    Mixing raises the output level: RMS(s+n) = RMS(s) * sqrt(1 + 10^(-SNR/10)),
    i.e. +0.5% at 20 dB but +41% at 0 dB and +100% at -5 dB. Left uncorrected,
    output level co-varies with SNR, and any VAD with a fixed absolute
    threshold or level-dependent front end (energy, WebRTC) would show an
    "SNR effect" that is partly a level effect. That is an undeclared channel
    confound in a study whose entire claim is that channel is held constant.

    `match_speech_rms` applies a single global gain so output RMS equals the
    original active-speech RMS. Because the gain is global it is applied to
    speech and noise alike, so **the achieved SNR is exactly invariant** --
    only the digital level changes. Gain-invariant systems (Silero, pyannote,
    which normalise internally) see no difference at all; gain-sensitive ones
    are no longer penalised for an artefact of mixing.

    Set `none` to reproduce the raw additive behaviour.
    """
    speech = np.asarray(speech, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)[: len(speech)]
    if len(noise) < len(speech):
        noise = np.pad(noise, (0, len(speech) - len(noise)))

    s_rms = active_rms(speech, speech_mask_samples)
    n_rms = rms(noise)
    if n_rms < EPS:
        return speech.copy()

    target_n_rms = s_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (target_n_rms / n_rms)
    mixed = speech + scaled_noise

    if level_normalization == "match_speech_rms":
        r_out = rms(mixed)
        if r_out > EPS:
            mixed = mixed * (s_rms / r_out)
    elif level_normalization not in ("none", None):
        raise ValueError(f"unknown level_normalization '{level_normalization}'")

    if verify:
        # Verify against the KNOWN components, not (mixed - speech): after a
        # global gain g the residual is (g-1)*speech + g*noise, which is not
        # the noise and would give a wrong answer.
        g = 1.0
        if level_normalization == "match_speech_rms":
            r_pre = rms(speech + scaled_noise)
            g = (s_rms / r_pre) if r_pre > EPS else 1.0
        achieved = 20.0 * np.log10(
            max(active_rms(g * speech, speech_mask_samples), EPS)
            / max(rms(g * scaled_noise), EPS))
        if abs(achieved - snr_db) > tolerance_db:
            print(f"[audio] SNR mismatch: requested {snr_db:+.1f} dB, "
                  f"achieved {achieved:+.1f} dB")

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        # Only relevant if written to 16-bit PCM; in-memory inference is fine.
        print(f"[audio] note: peak {peak:.2f} exceeds full scale "
              f"(SNR {snr_db} dB); clipping only occurs if written to wav")
    return mixed.astype(np.float32)


def apply_rir(x, rir, compensate_delay=True):
    """Convolve with a room impulse response.

    The direct-path delay is compensated so reference boundaries remain valid;
    reverberant tails still smear offsets, which we report as a known limitation
    rather than hide.
    """
    x = np.asarray(x, dtype=np.float32)
    rir = np.asarray(rir, dtype=np.float32)
    if rir.ndim > 1:
        rir = rir[:, 0]
    if len(rir) == 0 or not np.any(np.isfinite(rir)):
        return x.copy()
    peak = int(np.argmax(np.abs(rir))) if compensate_delay else 0
    y = fftconvolve(x, rir, mode="full")[peak:peak + len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    # Preserve input level so SNR bookkeeping stays interpretable.
    r_in, r_out = rms(x), rms(y)
    if r_out > EPS:
        y = y * (r_in / r_out)
    return y.astype(np.float32)


def make_babble(n_samples, excerpts, n_talkers, rng):
    """Synthesize babble by summing excerpts from held-out control languages."""
    if not excerpts:
        return rng.standard_normal(n_samples).astype(np.float32) * 0.01
    out = np.zeros(n_samples, dtype=np.float32)
    idx = rng.integers(0, len(excerpts), size=n_talkers)
    for i in idx:
        seg = tile_to_length(n_samples, excerpts[int(i)], rng)
        r = rms(seg)
        if r > 1e-6:
            out += (seg / r).astype(np.float32)
    r = rms(out)
    # NOTE: rms() floors at sqrt(EPS)=1e-5, so this guard must sit above that.
    if r <= 1e-4:
        # Degenerate excerpt set (all silent / too short). Fall back to noise
        # rather than silently emitting zeros, which would make every babble
        # condition secretly a clean condition.
        return rng.standard_normal(n_samples).astype(np.float32)
    return (out / r).astype(np.float32)


# --------------------------------------------------------------------------- #
# Framing / spectra
# --------------------------------------------------------------------------- #
def frame_signal(x, frame_len, hop):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def log_power_spectrogram(x, sr, frame_ms=25, hop_ms=10, n_fft=512):
    fl = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    frames = frame_signal(x, fl, hop) * np.hanning(fl)[None, :]
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    return 10.0 * np.log10(spec + EPS)


def n_frames_for(n_samples, sr, frame_ms, hop_ms):
    fl = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    return max(1, 1 + (max(n_samples, fl) - fl) // hop)

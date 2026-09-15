"""VAD system wrappers.

Every system exposes the same contract:

    system.posterior(x: np.ndarray, sr: int) -> np.ndarray  # [0,1], 10 ms grid

Heavy dependencies are imported lazily inside `load()` so that a machine
missing torch can still run the signal-processing baselines end to end.

WHICH SYSTEMS ACTUALLY BENEFIT FROM A GPU
-----------------------------------------
  energy, ltsd, webrtc : none. Pure numpy / C. GPU is irrelevant.
  silero               : none, and possibly negative. It is a ~1 MB model fed
                         512-sample chunks in a sequential Python loop, so it
                         is latency-bound; per-chunk kernel launches cost more
                         than the compute saves. Deliberately pinned to CPU.
  marblenet, pyannote  : real 5-10x, BUT ONLY because the wrappers below batch
                         their windows. A one-window-at-a-time loop leaves the
                         GPU idle and is barely faster than CPU.

So enabling a GPU does not speed up the CPU-only plan; it makes the two strong
neural baselines affordable, which is a paper-quality argument, not a speed one.
"""
from __future__ import annotations

import numpy as np

from .audio import frame_signal, log_power_spectrogram, n_frames_for
from .decode import resample_posterior

FRAME_MS = 10
EPS = 1e-10
REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


class BaseVAD:
    name = "base"
    outputs_probability = True   # False => hard decisions only (AUC undefined)
    uses_gpu = False             # documentation for the preflight report

    def __init__(self, cfg=None, device="cpu", **kw):
        self.cfg = cfg or {}
        self.device = device
        self.kw = kw

    def load(self):
        return self

    def posterior(self, x, sr):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1. Energy (adaptive threshold on frame log-energy)
# --------------------------------------------------------------------------- #
@register("energy")
class EnergyVAD(BaseVAD):
    name = "energy"

    def posterior(self, x, sr):
        fl, hop = int(sr * 0.025), int(sr * FRAME_MS / 1000)
        fr = frame_signal(x, fl, hop)
        e = 10.0 * np.log10(np.mean(fr ** 2, axis=1) + EPS)
        floor = np.percentile(e, 10)
        ceil = np.percentile(e, 95)
        span = max(ceil - floor, 6.0)
        return np.clip((e - (floor + 0.35 * span)) / (0.35 * span), 0.0, 1.0
                       ).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. LTSD  (Ramirez et al., long-term spectral divergence)
# --------------------------------------------------------------------------- #
@register("ltsd")
class LTSDVAD(BaseVAD):
    name = "ltsd"

    def __init__(self, cfg=None, device="cpu", order=6, **kw):
        super().__init__(cfg, device=device, **kw)
        self.order = order

    def posterior(self, x, sr):
        S = log_power_spectrogram(x, sr, 25, FRAME_MS)      # (T, F) in dB
        P = 10.0 ** (S / 10.0)
        T = P.shape[0]
        N = self.order
        # Noise spectrum from the quietest 10% of frames.
        e = P.sum(axis=1)
        quiet = P[np.argsort(e)[: max(1, T // 10)]].mean(axis=0) + EPS
        lte = np.zeros_like(P)
        for t in range(T):
            lo, hi = max(0, t - N), min(T, t + N + 1)
            lte[t] = P[lo:hi].max(axis=0)
        ltsd = 10.0 * np.log10(np.mean(lte / quiet, axis=1) + EPS)
        lo_q, hi_q = np.percentile(ltsd, 10), np.percentile(ltsd, 95)
        span = max(hi_q - lo_q, 3.0)
        return np.clip((ltsd - (lo_q + 0.4 * span)) / (0.4 * span), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. WebRTC (GMM, legacy floor)
# --------------------------------------------------------------------------- #
@register("webrtc")
class WebRTCVAD(BaseVAD):
    name = "webrtc"
    outputs_probability = False

    def __init__(self, cfg=None, device="cpu", aggressiveness=2, **kw):
        super().__init__(cfg, device=device, **kw)
        self.aggr = int(aggressiveness)
        self._v = None

    def load(self):
        import webrtcvad
        self._v = webrtcvad.Vad(self.aggr)
        return self

    def posterior(self, x, sr):
        if self._v is None:
            self.load()
        if sr not in (8000, 16000, 32000, 48000):
            raise ValueError(f"webrtcvad needs 8/16/32/48 kHz, got {sr}")
        step = int(sr * 0.01)                     # 10 ms frames -> native grid
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        n = len(x) // step
        out = np.zeros(n, dtype=np.float32)
        for i in range(n):
            chunk = pcm[i * step * 2:(i + 1) * step * 2]
            if len(chunk) < step * 2:
                break
            out[i] = 1.0 if self._v.is_speech(chunk, sr) else 0.0
        # Light smoothing so a threshold sweep is still meaningful.
        k = np.ones(5, dtype=np.float32) / 5.0
        return np.convolve(out, k, mode="same").astype(np.float32)


# --------------------------------------------------------------------------- #
# 4. Silero v5 (MIT)
# --------------------------------------------------------------------------- #
@register("silero")
class SileroVAD(BaseVAD):
    name = "silero"

    def __init__(self, cfg=None, device="cpu", **kw):
        # Pinned to CPU by design; see module docstring.
        super().__init__(cfg, device="cpu", **kw)
        self._model = None

    def load(self):
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad(onnx=True)
        except Exception:
            import torch
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad",
                force_reload=False, onnx=False, trust_repo=True)
        return self

    def posterior(self, x, sr):
        import torch
        if self._model is None:
            self.load()
        win = 512 if sr == 16000 else 256          # Silero's fixed window sizes
        n_win = len(x) // win
        probs = np.zeros(n_win, dtype=np.float32)
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()
        with torch.no_grad():
            for i in range(n_win):
                chunk = torch.from_numpy(
                    np.ascontiguousarray(x[i * win:(i + 1) * win], dtype=np.float32))
                probs[i] = float(self._model(chunk, sr))
        n_target = n_frames_for(len(x), sr, 25, FRAME_MS)
        return resample_posterior(probs, n_target)


# --------------------------------------------------------------------------- #
# 5. rVAD-fast (unsupervised SP baseline)
# --------------------------------------------------------------------------- #
@register("rvad")
class RVADFast(BaseVAD):
    name = "rvad"
    outputs_probability = False

    def load(self):
        from rVADfast import rVADfast          # noqa: F401
        self._f = rVADfast()
        return self

    def posterior(self, x, sr):
        if not hasattr(self, "_f"):
            self.load()
        labels, _ = self._f(np.asarray(x, dtype=np.float64), sr)
        n_target = n_frames_for(len(x), sr, 25, FRAME_MS)
        return resample_posterior(np.asarray(labels, dtype=np.float32), n_target)


# --------------------------------------------------------------------------- #
# 6. NeMo MarbleNet -- BATCHED (this is what makes a GPU worth having)
# --------------------------------------------------------------------------- #
@register("marblenet")
class MarbleNetVAD(BaseVAD):
    name = "marblenet"
    uses_gpu = True

    def __init__(self, cfg=None, device="cpu", batch_size=256,
                 window_s=0.63, shift_s=0.02, **kw):
        super().__init__(cfg, device=device, **kw)
        self.batch_size = int(batch_size)
        self.window_s = float(window_s)
        self.shift_s = float(shift_s)   # 20 ms, then upsampled to the 10 ms grid

    def load(self):
        import torch
        import nemo.collections.asr as nemo_asr
        self._m = nemo_asr.models.EncDecClassificationModel.from_pretrained(
            "vad_multilingual_marblenet")
        self._m.eval()
        self._m = self._m.to(torch.device(self.device))
        return self

    def posterior(self, x, sr):
        import torch
        if not hasattr(self, "_m"):
            self.load()
        win, hop = int(self.window_s * sr), int(self.shift_s * sr)
        xs = np.pad(x, (0, max(0, win - len(x))))
        n = max(1, 1 + (len(xs) - win) // hop)

        # Strided view -> one big (n, win) batch, sliced into GPU-sized chunks.
        idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
        windows = xs[idx].astype(np.float32)

        dev = torch.device(self.device)
        probs = np.empty(n, dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n, self.batch_size):
                b = torch.from_numpy(windows[i:i + self.batch_size]).to(dev)
                ln = torch.full((b.shape[0],), b.shape[-1],
                                dtype=torch.long, device=dev)
                logits = self._m.forward(input_signal=b, input_signal_length=ln)
                p = torch.softmax(logits, dim=-1)[:, 1]
                probs[i:i + len(p)] = p.detach().cpu().numpy()
        return resample_posterior(probs, n_frames_for(len(x), sr, 25, FRAME_MS))


# --------------------------------------------------------------------------- #
# 7. pyannote segmentation -- BATCHED (needs HF token + accepted model terms)
# --------------------------------------------------------------------------- #
@register("pyannote")
class PyannoteVAD(BaseVAD):
    name = "pyannote"
    uses_gpu = True

    def __init__(self, cfg=None, device="cpu", hf_token="", batch_size=32, **kw):
        super().__init__(cfg, device=device, **kw)
        self.hf_token = hf_token
        self.batch_size = int(batch_size)

    def load(self):
        import torch
        from pyannote.audio import Inference, Model
        if not self.hf_token:
            raise RuntimeError(
                "pyannote requires credentials.hf_token in config.yaml AND "
                "acceptance of the model terms at "
                "huggingface.co/pyannote/segmentation-3.0")
        m = Model.from_pretrained("pyannote/segmentation-3.0",
                                  use_auth_token=self.hf_token)
        self._inf = Inference(m, step=0.1, duration=5.0,
                              batch_size=self.batch_size,
                              device=torch.device(self.device))
        return self

    def posterior(self, x, sr):
        import torch
        if not hasattr(self, "_inf"):
            self.load()
        wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        out = self._inf({"waveform": wav, "sample_rate": sr})
        data = np.asarray(out.data, dtype=np.float32)
        # segmentation-3.0 emits powerset speaker classes; class 0 == silence.
        p = 1.0 - data[:, 0] if data.ndim == 2 else data.ravel()
        return resample_posterior(p, n_frames_for(len(x), sr, 25, FRAME_MS))


# --------------------------------------------------------------------------- #
def build_system(name, cfg, device=None):
    if name not in REGISTRY:
        raise KeyError(f"unknown VAD system '{name}'. Known: {sorted(REGISTRY)}")
    spec = dict(cfg.get("systems", {}).get(name, {}))
    spec.pop("enabled", None)
    spec.pop("subsample", None)
    if device is None:
        from .config import resolve_device
        device = resolve_device(cfg)
    cls = REGISTRY[name]
    # Silero is pinned to CPU on purpose -- see the module docstring.
    spec["device"] = device if getattr(cls, "uses_gpu", False) else "cpu"
    if name == "pyannote":
        spec["hf_token"] = cfg.get("credentials", {}).get("hf_token", "")
    return cls(cfg=cfg, **spec)


def enabled_systems(cfg):
    return [n for n, s in cfg.get("systems", {}).items() if s.get("enabled", False)]

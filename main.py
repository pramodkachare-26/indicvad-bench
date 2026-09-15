#!/usr/bin/env python3
"""IndicVAD-Bench -- pipeline orchestrator.

Usage
-----
  python main.py                       # run every step in order
  python main.py --steps 01 02 03      # run a subset
  python main.py --steps 04 --overwrite
  python main.py --dry-run             # print the plan and exit
  python main.py --steps 03b                       # human-gold pack: run any
                                                    # time, even before step 01
  python main.py --steps 03b --score-human-gold    # after annotations return

Each step is idempotent and resumable: completed work is cached under
project.root and skipped unless --overwrite is passed.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from vadlib.config import (disk_report, load_config,  # noqa: E402
                           resolve_device)
from vadlib.notify import notify, notify_step  # noqa: E402

STEPS = [
    ("01", "download",    "Fetch FLEURS + OpenSLR-28 RIRs/noises"),
    ("02", "pool",        "Committee-verified speech excerpt pool"),
    ("03", "benchmark",   "Synthesize matched-condition sessions (exact GT)"),
    ("03b", "human_gold", "Human-gold pack (synth+real). RUN ANY TIME -- no "
                          "scoring dependency; real half needs no prior step"),
    ("04", "run_vads",    "Run VAD systems over the condition grid"),
    ("04b", "ssl_probe",  "OPTIONAL frozen SSL probe, leave-one-language-out"),
    ("05", "score",       "Score posteriors -> frame_metrics.csv"),
    ("06", "stats",       "Variance decomposition (RQ1) + significance"),
    ("07", "hparams",     "Decoding-hyperparameter retuning (RQ3-lite)"),
    ("08", "phonology",   "Phonological density regression (RQ2)"),
    ("09", "ssl_features", "OPTIONAL frozen-SSL features + layer probe (GPU)"),
    ("10", "transfer",     "OPTIONAL RQ3: adaptation curves, LOLO/LOFO"),
    ("11", "paper_pack",   "Assemble CSVs, tables, figures for upload"),
]

MODULES = {
    "01": "steps.step01_download",
    "02": "steps.step02_pool",
    "03": "steps.step03_benchmark",
    "03b": "steps.step03b_human_gold",
    "04": "steps.step04_run_vads",
    "04b": "steps.step04b_ssl_probe",
    "05": "steps.step05_score",
    "06": "steps.step06_stats",
    "07": "steps.step07_hparams",
    "08": "steps.step08_phonology",
    "09": "steps.step09_ssl_features",
    "10": "steps.step10_transfer",
    "11": "steps.step11_paper_pack",
}


def preflight(cfg):
    print("=" * 74)
    print("IndicVAD-Bench preflight")
    print("=" * 74)
    device = resolve_device(cfg)
    print(f"  python      : {platform.python_version()} on {platform.system()}")
    print(f"  storage mode: {cfg['storage_mode']}")
    print(f"  archives    : {cfg['paths']['archives']}  (persistent)")
    print(f"  scratch     : {cfg['paths']['scratch_root']}  (ephemeral)")
    print(f"  results     : {cfg['paths']['results']}  (persistent)")
    print(f"  device      : {device}")
    if device == "cuda":
        try:
            import torch
            print(f"  gpu         : {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    for k, v in disk_report(cfg).items():
        print(f"  disk {k:12s}: {v['free_gb']:.1f} GB free "
              f"of {v['total_gb']:.1f} GB  ({v['path']})")
    print(f"  languages   : {len(cfg['languages'])} "
          f"({sum(1 for l in cfg['languages'] if not l.get('control'))} under test)")
    print(f"  sessions    : {cfg['benchmark']['sessions_per_lang']} per language")
    n_cond = (len(cfg["conditions"]["snr_db"]) *
              len(cfg["conditions"]["noise_types"]) *
              len(cfg["conditions"]["reverb"])) + 1
    print(f"  conditions  : {n_cond}")
    enabled = [k for k, v in cfg["systems"].items() if v.get("enabled")]
    print(f"  systems     : {enabled or 'NONE -- enable some in config.yaml'}")
    if device == "cuda":
        from vadlib.systems import REGISTRY
        gpu_sys = [s for s in enabled
                   if getattr(REGISTRY.get(s), "uses_gpu", False)]
        if not gpu_sys:
            print("  NOTE: a GPU is available but no enabled system uses it.")
            print("        energy/ltsd/webrtc/silero are CPU-bound by design.")
            print("        Enable marblenet and/or pyannote to get value from it.")
        else:
            print(f"  gpu-backed  : {gpu_sys}")

    n_test = sum(1 for l in cfg["languages"] if not l.get("control"))
    hours = (n_test * cfg["benchmark"]["sessions_per_lang"] *
             cfg["benchmark"]["session_duration_s"] * n_cond / 3600)
    print(f"  audio to score per system: {hours:.1f} h")
    print(f"  ==> rough budget: {hours/300:.1f} h for energy/ltsd/webrtc, "
          f"{hours/30:.1f} h for silero (CPU),")
    print(f"      {hours/(120 if device=='cuda' else 12):.1f} h for "
          f"marblenet/pyannote on {device}")
    print("      Measured throughput is printed per condition during step 04.")

    missing = []
    for mod in ("numpy", "scipy", "pandas", "yaml"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit(f"  MISSING required packages: {missing}\n"
                 f"  run: pip install -r requirements.txt")
    for opt, why in (("huggingface_hub", "step01 FLEURS download"),
                     ("torch", "silero / marblenet / pyannote / step09"),
                     ("webrtcvad", "webrtc baseline"),
                     ("statsmodels", "mixed-effects model in step06"),
                     ("matplotlib", "figures in step10")):
        try:
            __import__(opt)
        except ImportError:
            print(f"  [optional missing] {opt:16s} -- needed for {why}")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="IndicVAD-Bench pipeline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--steps", nargs="*", default=None,
                    help="subset of step ids, e.g. --steps 04 05 06")
    ap.add_argument("--overwrite", action="store_true",
                    help="ignore caches and recompute")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-human-gold", action="store_true",
                    help="in step 08, score returned human annotations")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.overwrite:
        cfg["runtime"]["overwrite"] = True

    preflight(cfg)

    todo = args.steps if args.steps else [s[0] for s in STEPS]
    todo = [t if t.endswith("b") else t.zfill(2) for t in todo]

    print("\nPlan:")
    for sid, name, desc in STEPS:
        mark = "->" if sid in todo else "  "
        print(f" {mark} [{sid}] {name:12s} {desc}")
    if args.dry_run:
        return 0

    log_path = Path(cfg["paths"]["logs"]) / \
        f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    print(f"\nlog: {log_path}\n")
    t_all = time.time()
    failures = []

    with open(log_path, "w", encoding="utf-8") as log:
        for sid, name, desc in STEPS:
            if sid not in todo:
                continue
            header = f"\n{'='*74}\n[{sid}] {name} -- {desc}\n{'='*74}"
            print(header)
            log.write(header + "\n")
            t0 = time.time()
            try:
                mod = __import__(MODULES[sid], fromlist=["run"])
                if sid == "03b":
                    out = mod.run(cfg, score_human=args.score_human_gold)
                else:
                    out = mod.run(cfg)
                dt = time.time() - t0
                msg = f"[{sid}] OK in {dt/60:.1f} min -> {out}"
                print(msg); log.write(msg + "\n")
                notify_step(cfg, sid, True, dt / 60, f"-> {out}")
            except SystemExit as e:
                msg = f"[{sid}] HALTED: {e}"
                print(msg); log.write(msg + "\n")
                failures.append(sid)
                if not args.continue_on_error:
                    break
            except Exception:
                tb = traceback.format_exc()
                print(f"[{sid}] FAILED:\n{tb}")
                log.write(f"[{sid}] FAILED:\n{tb}\n")
                notify_step(cfg, sid, False, (time.time() - t0) / 60,
                            tb.strip().splitlines()[-1][:180])
                failures.append(sid)
                if not args.continue_on_error:
                    break

    total = (time.time() - t_all) / 60
    print(f"\n{'='*74}")
    print(f"finished in {total:.1f} min; failures: {failures or 'none'}")
    print(f"paper pack: {cfg['paths']['paper']}")
    print("=" * 74)
    notify(cfg,
           "indicvad finished" if not failures else "indicvad finished WITH ERRORS",
           f"{total:.1f} min total. failures: {failures or 'none'}. "
           f"Paper pack: {cfg['paths']['paper']}",
           priority="high" if failures else "default",
           tags="tada" if not failures else "rotating_light")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

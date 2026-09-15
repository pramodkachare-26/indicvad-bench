"""Step 03b -- human-gold annotation pack. Standalone, no scoring dependency.

WHY THIS IS ITS OWN STEP
-------------------------
The pack has two halves with very different dependencies:

  real (IndicVoices)  -- needs only config.yaml + network. Nothing else in
                         this pipeline has to have run. Not even Step 01.
  synthetic            -- needs `sessions.csv` from Step 03.

Neither half has anything to do with scoring. The annotation pack used to be
generated at the end of Step 08 (phonology), which meant it was unreachable
until Steps 01-07 had all completed -- and Step 08 itself crashed on a
missing `frame_metrics.csv` before ever reaching it. That is backwards: the
one part of this pipeline that depends on a human's calendar rather than
compute should be kicked off FIRST, in parallel with everything else, not
queued behind seven other steps.

RUN THIS ANY TIME. Recommended: run it on day 1, before Step 01, to start the
real-audio download immediately. Re-run after Step 03 to pick up the
synthetic half once sessions exist -- already-fetched real clips are not
re-downloaded (`--overwrite` forces a refresh).

    python main.py --steps 03b                    # kick off real download,
                                                    # synthetic half is empty
                                                    # until Step 03 has run
    python main.py --steps 01 02 03                # build the benchmark
    python main.py --steps 03b                     # re-run: synthetic half
                                                    # now fills in too
    python main.py --steps 03b --score-human-gold   # after annotations return
"""
from __future__ import annotations

from steps.step08_phonology import make_human_gold_pack, score_human_gold


def run(cfg, score_human=False):
    out = make_human_gold_pack(cfg)
    if score_human:
        score_human_gold(cfg)
    return out

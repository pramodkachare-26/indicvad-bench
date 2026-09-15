# IndicVAD-Bench

**Disentangling language from channel in voice activity detection for Indian languages.**

Target venue: **ICASSP 2027** — Toronto, 16–21 May 2027. Paper deadline
**16 September 2026, 23:59:59 AoE (UTC−12)**. 4 pages + 1 page references.

---

## 1. What this answers, and what it deliberately does not

The original framing — *"is there language dependence in state-of-the-art
VAD?"* — is not identifiable. Run any VAD on Hindi/IndicVoices and
Tamil/SPRING-INX, get different numbers, and you have measured a **corpus**
effect: different microphones, rooms, SNR distributions, elicitation styles and
annotation conventions. Not language.

This pipeline makes the question answerable by holding everything else fixed:

| Confound | How it is neutralised |
|---|---|
| Content | **FLEURS** is the speech version of FLoRes — the *same sentences* in every language |
| Recording condition | Single collection protocol across all FLEURS languages |
| Pause structure | Silences inserted by us, seeded on **session index only, never language** |
| Noise / room | Identical noise waveform and RIR in a given (session, condition) cell across all languages |
| Label protocol | One convention, applied identically; ground truth exact by construction |

The three questions actually asked:

* **RQ1 (decomposition).** With content, channel, noise and speaker held
  constant, what share of VAD error variance is attributable to language
  identity versus SNR, noise type and reverberation? → `variance_components.csv`
* **RQ2 (localisation).** Does error concentrate where Indic phonology predicts —
  breathy-voiced stops, aspirates, geminate closures? → `phono_regression.csv`
* **RQ3-lite (remedy economics).** Is any residual language gap closable by
  retuning **three decoding scalars**, with no retraining? → `hparam_tuning.csv`

### Scope: what is in the ICASSP paper

**13 languages under test + 1 control, all from FLEURS.** One corpus, one
recording protocol — adding a second speech corpus would reintroduce the exact
confound this study exists to remove. More corpora belong in the journal paper,
with forced-alignment pseudo-GT and a proper external-validity section.

| Family | Languages |
|---|---|
| Indo-Aryan (9) | Hindi, Bengali, Marathi, Gujarati, Punjabi, Odia, Assamese, Nepali, Urdu |
| Dravidian (4) | Tamil, Malayalam, Telugu, Kannada |
| Control (babble source only) | English (en_us) |

**On the 9:4 family split.** FLEURS has no further Dravidian languages to add,
so the imbalance cannot be fixed by adding more South Asian languages within
this corpus. Rather than forcing a two-group Indo-Aryan-vs-Dravidian test that
an imbalanced 9:4 split would make easy to over-read, **`family` is reported
only as a secondary, explicitly-labelled descriptive aggregate**
(`family_effects_secondary.csv`, `role=secondary_descriptive_not_headline`).
RQ1's actual claim is the **per-language** variance decomposition in
`variance_components.csv` — 13 levels, not 2 — which is unaffected by the
family split and is the correct headline result. Step 06 prints a warning if
the family ratio exceeds 1.5x, as a reminder not to over-read the secondary
table.

**Framing:** Urdu (PK) and Nepali (NP) are not Indian languages. The title and
abstract say **South Asian**.

**Corpora: 2.** FLEURS (speech) + OpenSLR-28 (RIRs and point-source noises).
MUSAN is optional and off by default. Babble is synthesised from the control
language, so it never contains a language under test.

**Conditions: 37.** 6 SNR × 3 noise types × 2 reverb, plus a clean anchor.

### Honest scope limits

* **Read speech, re-concatenated.** Natural conversational pause distributions
  are *not* measured — we control pauses deliberately. State this in §3.
* **No Tibeto-Burman arm.** The original design had Manipuri/Mizo as the
  maximal-typological-distance contrast. FLEURS has no Tibeto-Burman South
  Asian language, so this cannot be recovered within the matched-corpus design
  at any compute budget. Declare it.
* **No code-switching arm.** H4 needed DISPLACE, which is excluded (see below).
  Untestable here.
* **Urdu is detection-only.** The Step 08 density proxy reads ISCII-derived
  Unicode blocks; Perso-Arabic does not parse. Urdu contributes to RQ1/RQ2
  detection metrics but is excluded from the phonology regression. This is a
  shame — Hindi/Urdu is a near-minimal pair (shared phonology, different
  script) — but the orthographic proxy cannot exploit it.
* **Orthographic phonology proxy**, not forced alignment. Label it as such.
* **No DISPLACE, no Vaani.** DISPLACE requires a signed per-edition Terms &
  Conditions and forbids redistribution; Vaani is form-gated with unclear
  licence terms. Both are excluded so the pipeline carries zero access risk.
* **Full RQ3 is deferred.** Adaptation curves, leave-one-family-out and
  transfer geometry are implemented (Steps 09–10) but disabled by default —
  they are the journal paper. With 13 languages, regressing transfer gain on
  typological distance is underpowered regardless of hardware; that needs
  ~20+ languages.

### The two vulnerabilities you must manage

1. **Synthetic-only is the weakest flank.** A reviewer will say concatenated
   read speech is not VAD. Step 08 emits a human-annotation pack **split
   50/50 between synthetic benchmark sessions and real spontaneous audio from
   IndicVoices** (`human_gold.split_synthetic_frac`, default 0.5). Synthetic
   ground truth is already exact by construction, so annotating it only
   validates the committee's excerpt boundaries; the real half buys
   speech-style external validity with no forced alignment (humans supply the
   label directly) at essentially no extra compute. Hand-labelling ~150 s per
   language (split across both halves) is the highest-value non-code task
   available. **Report the two subsets separately** in the paper —
   `human_gold_summary.csv` keeps a `source` column for exactly this reason;
   they answer different questions and must not be pooled into one number.
2. **Circularity.** References are built by a VAD committee that includes
   systems you then evaluate. Mitigations: 150 ms edge erosion,
   `reference.leave_system_out`, and now **Fleiss' κ reporting** (below).

## 2. Reference construction and agreement (κ)

Reference labels are **not** an ITU standard. ITU-T G.191/G.192 define how to
*score* a VAD given a reference; they do not say how to build one. Claiming
"ITU-compliant reference" would be false.

What the pipeline actually does, and what §3 should say:

> Speech regions are identified by committee: energy, LTSD and Silero run
> independently on each utterance; only frames with unanimous agreement are
> retained; boundaries are eroded inward by 150 ms to exclude edge smearing;
> interior excerpts ≥1 s are kept. Sessions are then assembled from these
> excerpts with silences inserted by the experimenter, so ground truth is exact
> by construction. Internal gaps shorter than 200 ms remain labelled speech
> (stop closures, geminates) — the convention the human annotation guide also
> follows.

Step 02 reports **Fleiss' κ** across the committee plus **pairwise Cohen's κ**,
per language, to `committee_agreement.csv`. This pre-empts the obvious
question — *did the detectors actually agree, or did unanimity simply never
trigger?* — and exposes an outlier member through the pairwise values.

Interpretation (Landis & Koch): >0.80 almost perfect, 0.61–0.80 substantial.
Step 02 warns if mean κ < 0.60, which would mean the unanimity rule is doing
more work than the detectors are. Report κ in §3 alongside the
`leave_system_out` sensitivity result.

### What exactly counts as "speech" — a worked example

The operational definition is narrower than what a human transcriber would
call speech, and it is worth tracing end to end so the limitation is concrete
rather than abstract.

**Input:** a Hindi FLEURS recording, *"भारत में मानसून जून में आता है"*
(~3.2 s, studio-clean, one continuous read sentence).

**Step 1 — committee vote (Step 02).** Energy, LTSD and Silero each label
every 10 ms frame independently. They agree on the interior; they disagree
most right at onset/offset, where energy rises gradually and each detector
draws the line a few tens of ms differently:

```
time (s):   0.0        0.18                          2.95    3.2
            |-----------|-----------------------------|-------|
energy:     silence     SPEECH.......................  silence
LTSD:       silence      SPEECH......................   silence
silero:     silence       SPEECH.....................    silence
                          ^ unanimous agreement starts   ends ^
```

Only frames where **all three** agree are kept; the fringe where they
disagree is discarded.

**Step 2 — erosion.** The unanimous region `[0.18, 2.95]` is shrunk inward by
150 ms on each side, giving `[0.33, 2.80]`. This buffer exists precisely
because the committee disagrees most near real boundaries, so eroding
discards the ambiguous edge and keeps only the interior everyone was
confident about.

**Step 3 — extract.** The waveform from 0.33 s to 2.80 s (2.47 s) becomes its
own excerpt file, treated as **100% speech with no internal structure**.

**Step 4 — reassemble (Step 03).** Excerpts from unrelated FLEURS sentences
are concatenated with silence **inserted by us**, duration drawn from a
lognormal (median ~0.5 s, range 0.25–2.5 s). Ground truth is then pure
arithmetic — we know the boundaries because we placed them:

```
speech_intervals = [(1.0, 3.47), (4.15, 6.05), (7.15, 10.15), ...]
```

**The operational definition, stated plainly:**

> *"Speech"* = the interior (minus 150 ms per edge) of a FLEURS utterance
> where energy, LTSD and Silero unanimously agreed it was speech.
> *"Silence"* = audio inserted afterward by the experimenter, or the eroded
> fringe the committee disagreed about.

That is a real, precise definition. It is just narrower, and different from
what a human listening to natural speech would call speech.

### Is this defensible?

Yes for some claims, no for others — say both in §3 rather than only one.

**Defensible:**

| Claim | Why |
|---|---|
| Relative system ranking under known SNR/reverb/language | Every system faces identical corrupted audio and an identical boundary definition — a fair race, even with a narrow finish line. |
| RQ1: variance decomposition (language vs. SNR vs. noise) | Content and pause structure are held constant, so differences across languages still reflect real acoustic/phonological properties, not annotation drift. |
| RQ2: boundary precision (onset/offset deviation, over-splitting) | The true boundary is known up to committee/erosion uncertainty, so millisecond-level disagreement is meaningful. |

**Not defensible:**

| Gap | What's missing |
|---|---|
| Real onsets/offsets | Soft onsets, trailing vocal fry, breath before a word — the hardest part of real VAD — is exactly the 150 ms fringe eroded away. The difficult cases are excluded by construction. |
| Disfluency, filled pauses, overlap | FLEURS is clean read speech: no "um," no cross-talk, no whispering, no emotional register. |
| Natural pause statistics | Real conversational pauses are not lognormal with these parameters, and plausibly correlate with language — the very confound this design avoids, at the cost of the silences being unrepresentative of what a real VAD encounters. |
| Circularity | A bias shared by all three committee members (e.g. consistently late onset on retroflex-initial syllables) is baked into "ground truth"; a system that gets the *true* onset right is then scored as wrong. |

**One-line summary for §3:**

> Reference labels are the committee-verified, edge-eroded interior of clean
> read-speech utterances; onsets, offsets and pause structure are synthetic
> by construction. This isolates language and channel effects cleanly but
> excludes the boundary and disfluency phenomena that dominate real-world VAD
> difficulty — addressed only by the human-gold real-audio subset (§3 below),
> which should be read as a separate, weaker-N sanity check, not a validation
> of the synthetic numbers.

This is also why the **real-IndicVoices half** of the human-gold pack matters
more than its size suggests: it is the only part of the pipeline that tests
VAD on natural onsets, offsets and pause structure at all. The synthetic
benchmark, however carefully built, answers a narrower and cleaner question
than "does this VAD work on real South Asian speech."

---

## 3. Human-gold annotation pack: standalone step, start it on day 1

**This is `python main.py --steps 03b` — a standalone step, not part of
scoring.** It used to live at the end of Step 08 (phonology), which meant it
was unreachable until Steps 01–07 had all finished, and Step 08 itself would
crash on a missing `frame_metrics.csv` before ever getting there. That was
backwards: annotation is the one part of this pipeline bound by a human's
calendar rather than compute, so it should start **first**, in parallel with
everything else — not queued behind seven other steps.

**Run it before Step 01 even exists on disk:**

```bash
python main.py --steps 03b          # kick off the real-audio download today
```

`step03b_human_gold.py` (`make_human_gold_pack` / `score_human_gold`, defined
in `step08_phonology.py` and imported from there) emits two subsets into
`03_benchmark/human_gold/`:

| Subset | Source | Ready when | Validates |
|---|---|---|---|
| Real (`real/*.wav`) | **IndicVoices** (`ai4bharat/IndicVoices`, CC BY 4.0), streamed | **Immediately** — needs only `config.yaml` + network | External validity on real spontaneous speech |
| Synthetic (top-level `*.wav`) | Benchmark test sessions | After Step 03 | Committee excerpt-boundary quality |

Run it again after Step 03 finishes and the synthetic half fills in
automatically — already-fetched real clips are not re-downloaded.

Real clips are pulled via `datasets.load_dataset(..., streaming=True)` — this
reads shards lazily and stops once each language's budget
(`human_gold.seconds_per_language * (1 - split_synthetic_frac)`) is met, so it
never downloads the full multi-thousand-hour corpus.

**Before running:** `human_gold.real_source.lang_config_map` maps each FLEURS
code to an IndicVoices HF config name (lowercase full English name, e.g.
`hindi`, `bengali`). Only `assamese` and `bengali` were directly confirmed
against the dataset card at write time — **verify the rest against
https://huggingface.co/datasets/ai4bharat/IndicVoices before relying on them**.
IndicVoices also asks for an active HF token even though the data itself is
CC BY 4.0 (dataset-viewer gating, not a licence restriction) — reuses
`credentials.hf_token`.

**Failure handling is per-language and per-half, not all-or-nothing.** A wrong
config name, missing `datasets` package, no network, or `sessions.csv` not
existing yet all degrade gracefully — each produces a clear note in
`INSTRUCTIONS.txt` (and, for the synthetic half, a console message explaining
it's expected pre-Step-03) rather than failing the whole pipeline. Re-run
`python main.py --steps 03b` after fixing access or completing Step 03;
completed clips are not re-fetched.

```bash
python main.py --steps 03b --score-human-gold   # after annotations return
```

keeps `source` (`synthetic` / `real_indicvoices`) in `human_gold_summary.csv`
so the two subsets are never silently averaged together.

---

## 4. Data and licences (all verified)

| Source | Licence | Access |
|---|---|---|
| FLEURS | **CC BY 4.0** | Public, instant |
| OpenSLR-28 (RIRs + point-source noises) | Free research release | Public, 1.3 GB |
| MUSAN *(optional)* | CC / US Public Domain, commercial OK | Public, 11 GB |

Chosen deliberately: OpenSLR-28 supplies **both** impulse responses **and**
noises (extracted from MUSAN) in one 1.3 GB download, so the default run needs
no 11 GB fetch. Babble is synthesised from control-language excerpts, so it
never contains a language under test.

`config.yaml` also documents IndicVoices, SPRING-INX and IndicTTS (all CC BY
4.0) if you later want to extend beyond FLEURS.

---

## 5. Where to run it

### Colab GPU (recommended) — `colab_indicvad.ipynb`

Open the notebook, set **Runtime → Change runtime type → T4 GPU**, and work
through the cells. It handles Drive mounting, the storage split, install,
self-test, a smoke run, and the full pipeline.

**Be clear about what the GPU buys you.** It does *not* speed up the default
system set:

| System | GPU benefit | Why |
|---|---|---|
| energy, ltsd | none | pure numpy |
| webrtc | none | C GMM, CPU-native |
| silero | **none, possibly negative** | ~1 MB model fed 512-sample chunks in a sequential loop; latency-bound, so per-chunk kernel launches cost more than they save. Pinned to CPU on purpose. |
| **marblenet** | **5–10×** | only because the wrapper batches windows |
| **pyannote** | **5–10×** | 5 s windows, batched transformer |
| step 09 tiny CNN | ~10× | batched training |

So the reason to use a GPU is **not** speed on the existing plan — it is that
`marblenet` and `pyannote` become affordable, and adding two strong neural
baselines materially strengthens §4. If you enable neither, run on CPU and
lose nothing. `main.py --dry-run` warns you if a GPU is present but idle.

Rough wall-clock for the default 8-language × 20-session × 37-condition grid
(~74 h of audio per system):

| Systems enabled | CPU (4 cores) | Colab T4 |
|---|---|---|
| energy + ltsd + webrtc | ~25 min | ~25 min |
| + silero | ~3 h | ~3 h |
| + pyannote + marblenet | ~14 h | **~4 h** |
| + full RQ3 (steps 09-10) | ~8 h more | **~1 h more** |

Steps 05–10 run on cached posteriors and take ~30 min either way.

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional
```

Set `storage.mode: local` in `config.yaml` (or leave `auto` — it detects).

---

## 6. Storage layout (read this before running on Colab)

Google Drive is a FUSE mount: fine for a handful of large files, pathological
for tens of thousands of small ones. Step 02 alone writes ~20k excerpt WAVs.
Pointing everything at Drive turns a 30-minute step into hours and can trip
Drive API quotas mid-run. So paths are split by access pattern:

| Path | Location under `mode: colab` | Survives disconnect |
|---|---|---|
| `00_archives/` — FLEURS tarballs, RIRs zip | **Drive** | ✅ |
| `01_raw/` — extracted audio | local scratch | ❌ |
| `02_pool/`, `03_benchmark/`, `04_posteriors/` | local scratch | ❌ |
| `05_results/`, `06_paper_pack/`, `logs/` | **Drive** | ✅ |

Re-extracting from Drive archives after a session restart costs ~2 min.
Re-downloading costs 30–90 min. That is the whole trade.

**Recovery after a disconnect:**

```bash
python main.py --steps 01 02 03   # ~15 min, no re-download
python main.py --steps 04         # completed conditions are skipped
```

---

## 7. Validate before you burn hours

```bash
python selftest.py      # expect: 52 passed, 0 failed
```

Runs offline in ~20 s. Checks I/O, active-speech SNR mixing (including exact
SNR invariance under level normalisation), RIR delay compensation, decoding
constraints, every metric against known-answer cases, the variance
decomposition against a planted effect, committee-agreement kappa against
known noise levels, and the phonology extractor. Several real bugs were
caught this way during development — including one in the day-1 human-gold
workflow this run() re-verified above.

Then **smoke test before the full grid** (notebook cell 7, or trim
`config.yaml` to 3 languages / 2 SNRs / `energy`+`webrtc`). Confirm the CSVs
look sane, then scale up.

---

## 8. Run

```bash
python main.py --dry-run             # plan, device, disk, runtime budget
python main.py                       # everything, in order
python main.py --steps 04 05 06 07   # a subset
python main.py --steps 04 --overwrite
```

Every step is idempotent and resumable. Step 04 checkpoints per
(system, condition) and prints measured throughput plus a running ETA.

| Step | Does | Writes |
|---|---|---|
| 01 | Download FLEURS + OpenSLR-28 → Drive, extract → scratch | `01_raw/utterances.csv` |
| 02 | Committee-verified excerpt pool | `02_pool/excerpts_all.csv` |
| 03 | Synthesise sessions, exact GT | `03_benchmark/sessions.csv` |
| **03b** | **Human-gold pack (synth+real). Run any time — the real half needs no prior step at all** | `03_benchmark/human_gold/` |
| 04 | **Run VADs over the grid** | `04_posteriors/*/*.npz` |
| 04b | *Optional GPU* frozen SSL probe, leave-one-language-out | `04_posteriors/ssl_probe/` |
| 05 | Score → master table | `frame_metrics.csv` |
| 06 | Variance decomposition (RQ1) | `variance_components.csv` +4 |
| 07 | Decoding retuning (RQ3-lite) | `hparam_tuning.csv` |
| 08 | Phonology (RQ2) | `phono_regression.csv` |
| 09 | *Optional* frozen-SSL features + layer probe (GPU) | `ssl_layer_probe.csv` |
| 10 | *Optional* RQ3 transfer curves, LOLO/LOFO | `transfer_curves.csv` |
| 11 | Assemble upload bundle | `06_paper_pack/` |

Step 08 no longer emits the annotation pack — see §3. If `frame_metrics.csv`
is missing (Steps 01–05 haven't run), Step 08 prints a message pointing to
Step 03b and returns cleanly instead of crashing.

### The SSL probe is a separate category, not a peer row

Every other system is **zero-shot** — it never sees the benchmark before being
scored. Step 04b trains a frozen-SSL probe **leave-one-language-out**: the head
scoring Tamil is trained only on the other twelve languages and never sees a
Tamil frame. Report it in the systems table under its own heading —
*supervised in-domain, unseen target language* — not alongside Silero and
pyannote. Printing it as a peer would flatter it unfairly.

It answers two reviewer objections at once: "you only benchmarked weak
detectors" (it is a modern SSL system), and "how much headroom is there at
all?" (it approximates the ceiling reachable without target-language
supervision). `ssl_probe_meta.csv` records the encoder, layer, protocol and
category so the table caption writes itself.

If Step 09's feature cache exists, Step 04b reuses it instead of re-encoding.

### Monitoring from your phone

Set `credentials.ntfy_topic` to any hard-to-guess string, then open
`https://ntfy.sh/<that string>` on your phone. You get a push notification
after each step in `runtime.notify_steps` (default: 1, 4, 5, 10) and on any
failure. No signup, no API key. Delivery failures never kill the run.

```yaml
credentials:
  ntfy_topic: "indicvad-a7f3c091"
```

Or `export NTFY_TOPIC=...` — the environment overrides the file, so no secret
needs committing.

---

## 9. Full RQ3 (steps 09-10) — journal track, disabled by default

Enable with `ssl.enabled: true`. Requires a GPU to be practical.

**Why it is affordable.** The encoder is frozen and its features are cached
once by step 09. Every transfer experiment in step 10 then trains a small head
on cached tensors, so the whole grid of protocols x budgets x languages costs
minutes instead of GPU-hours.

| Component | Cost on T4 | Included |
|---|---|---|
| WavLM-base feature extraction (4-condition subset) | ~25 min once | yes |
| Layer-wise probe | ~10 min | yes |
| Adaptation curves {1, 5, 15, 60} min x 8 languages | ~30 min | yes |
| Leave-one-language-out + leave-one-family-out | included | yes |
| LoRA/adapter fine-tuning of the encoder | 20+ GPU-h | **no** |

**Protocols** per held-out target language L:

| Protocol | Trained on | Measures |
|---|---|---|
| `matched` | all languages incl. L | upper bound |
| `lolo` | the other 7 | unseen-language penalty |
| `lofo` | the other family only | typological penalty |
| `lolo+adapt(n)` | `lolo`, then n min of L | adaptation economics |

**Layer probe.** VAD information in SSL models peaks in early-to-middle layers,
not the last one that many papers take by default. `layer: auto` measures this
and reports `ssl_layer_probe.csv`. If the best layer is not the last, say so.

**Two result shapes to expect, and how to read them:**

* Gap closure **above 100%** means the adapted head beat the matched
  multilingual head. That is real -- a specialist can outperform a generalist.
  Report it; do not clamp it.
* The typology correlation may come back **undefined**. With a balanced
  two-family design every language sits at the same mean distance from the
  others, so the predictor has no variance. Step 10 detects this and says so
  rather than emitting a bare `nan`. Fixing it needs languages at varying
  typological remove, not two tight clusters.

### The binding constraint is page count, not compute

You have 4 pages. The original 1.4-page §5 budget already covers the variance
table, the phone-class figure and the hyperparameter table. Adding RQ3 costs
about 0.35 more.

| Choice | Page cost | Verdict |
|---|---|---|
| RQ3 as **one** figure (curves + LOLO/LOFO), typology -> one sentence | -0.35 | **recommended** |
| Compress §3, point to released code for construction detail | +0.25 | do this to pay for it |
| RQ3 with layer-probe figure *and* typology figure | -0.8 | does not fit |

---

## 10. SNR, level, and mixing

**SNR is active-speech SNR.** Speech level is measured over reference speech
frames only. Sessions are majority silence by construction (inserted pauses up
to 2.5 s), so whole-signal RMS would understate speech level and silently shift
the true SNR downward.

```
SNR_dB = 20 log10( RMS(speech over reference speech frames) / RMS(scaled noise) )
target_noise_rms = speech_rms / 10^(SNR_dB / 20)
mixed            = speech + noise * (target_noise_rms / noise_rms)
```

**Output level is then normalised** (`conditions.level_normalization:
match_speech_rms`). Mixing inflates level by `sqrt(1 + 10^(-SNR/10))` — +0.5% at
20 dB but **+41% at 0 dB and +100% at −5 dB**. Uncorrected, digital level
co-varies with SNR, and any VAD with a fixed absolute threshold or
level-dependent front end (energy, WebRTC) shows an "SNR effect" that is partly
a level effect. In a study whose entire claim is that channel is held constant,
that is an undeclared confound.

A single global gain is applied so output RMS equals the original active-speech
RMS. Because the gain is global it scales speech and noise alike, so **the
achieved SNR is exactly invariant** — verified to <0.01 dB in `selftest.py`.
Gain-invariant systems (Silero, pyannote, which normalise internally) are
unaffected; gain-sensitive ones are no longer penalised for an artefact of
mixing. Set `none` for raw additive mixing.

`verify_snr: true` recomputes the achieved SNR from the known components and
warns on >0.25 dB drift. Note it recomputes from components, *not* from
`output − input`: after a global gain that residual is `(g−1)·speech + g·noise`,
which is not the noise.

**Determinism.** Noise clip, RIR, and pause structure are seeded on
`(session_idx, cond_id)` and **never on language**, so session 7 at SNR 10 dB
receives the identical noise waveform and room in every language. This is what
licenses the claim that language is the only free factor. Do not remove it.

---

## 11. Configuration

Everything tunable is in **`config.yaml`**; no hyperparameters or secrets in
code. Keys you will actually touch:

* `storage.mode` — `auto` (detects Colab), `colab`, or `local`.
* `runtime.device` — `auto`, `cuda`, or `cpu`.
* `credentials.hf_token` — needed for pyannote. Get one at
  huggingface.co/settings/tokens **and accept the model terms** at
  huggingface.co/pyannote/segmentation-3.0, or the download fails.
  `HF_TOKEN` in the environment overrides the file.
* `credentials.ntfy_topic` — phone notifications; see above.
* `conditions.level_normalization` — `match_speech_rms` (default) or `none`.
* `reference.committee` — default `[energy, ltsd, silero]`. **Two is the
  minimum** for a meaningful κ; a single-member committee makes unanimity
  vacuous and Step 02 warns.
* `ssl_probe.enabled` — the ICASSP LOLO probe (Step 04b). GPU recommended.
* `ssl.enabled` / `transfer` — journal-track full RQ3 (Steps 09–10). Leave off
  for the ICASSP run.
* `human_gold.split_synthetic_frac` — default 0.5. Fraction of the annotation
  budget spent on synthetic sessions vs. real IndicVoices audio.
* `human_gold.real_source.lang_config_map` — FLEURS code → IndicVoices HF
  config name. Verify against the dataset card before running (see §3).
* `languages` — codes are FLEURS configs. `control: true` languages are held
  out of the benchmark and used as the babble source.
* `conditions` — the corruption grid; this drives total runtime linearly.
* `systems.*.enabled` / `.subsample` — subsample slow systems rather than
  disabling them outright.
* `benchmark.sessions_per_lang` — **keep ≥ 10.** With `dev_fraction: 0.4`,
  fewer than 3 dev sessions per language makes Step 07 overfit, and
  per-language tuning will look *worse* than global. Step 07 warns when this
  happens.
* `data.max_utts_per_lang` — **keep ≥ 200.** Step 08 needs many distinct
  transcripts per language, otherwise the density features have no
  within-language variation and are perfectly collinear with language identity.
  Step 08 detects this and refuses to report the affected rows.

---

## 12. Reading the output

Upload the whole **`run/06_paper_pack/`** folder. It contains the CSVs, a
LaTeX-ready `tables.tex`, two figures, and `SUMMARY.md` with the headline
numbers.

* **`variance_components.csv`** is the paper's central claim. If `lang` has a
  small ω² relative to `snr_db` and `reverb`, the honest conclusion is that VAD
  error across Indian languages is **channel-driven, not language-driven** —
  and the practical recommendation is to spend annotation budget on channel and
  noise diversity rather than per-language VADs. Report that plainly.
* **`permutation_test.csv`** guards against reading noise as a language effect.
  Check the p-value *before* claiming an effect in either direction.
* **`hparam_tuning.csv`** is the positive result that insures against a null in
  RQ1. Key columns: `rel_improvement`, `gap_closed_frac`, and whether
  `best_min_silence_s` shifts systematically by language (hypothesis H3:
  syllable-timed rhythm moves the optimal hangover).
* **`phono_regression.csv`** — only rows with `controls_language=True` **and**
  `collinearity_warning=False` are interpretable. A density effect that
  survives language dummies is a real phonological effect; one that does not is
  a relabelled language effect.
* **`family_effects_secondary.csv`** is descriptive only — do not lead the
  paper with an Indo-Aryan-vs-Dravidian contrast; the 9:4 split is
  underpowered for that as a standalone claim. Cite it as supporting context
  next to the per-language decomposition, not in place of it.
* **`committee_agreement.csv`** justifies the reference. Quote mean Fleiss' κ
  in §3. If it is below 0.60, the reference is weak and you should lean harder
  on the human gold set.
* **`ssl_probe_meta.csv`** documents the probe's protocol so it is reported as
  a separate category, not a peer row.
* **`human_gold_summary.csv`** is your external-validity anchor. If it is
  absent, say so in the limitations rather than implying the synthetic
  benchmark was validated against human labels.

### If RQ1 comes back null

That is a likely and publishable outcome — *provided* it is framed as a
benchmark-and-analysis contribution with an actionable recommendation, paired
with the Step 07 positive result and the Step 08 localisation. It is not
publishable as "we tested and found nothing." Decide by the **Sep 6 gate**: if
the decomposition is inconclusive *and* Step 07 shows no meaningful gap
closure, pivot to the benchmark-and-protocol contribution (the matched-condition
design plus standardised causal/non-causal evaluation) rather than forcing a
claim.

---

## 13. Suggested 4-week schedule

| Window | Milestone |
|---|---|
| **Aug 20 (day 1)** | **`python main.py --steps 03b`** — start the real-audio download and commission the annotator *before* anything else, in parallel with everything below |
| Aug 20–23 | Colab setup, `selftest.py`, smoke run, Step 01–03 |
| Aug 24–30 | Full Step 04 on GPU with pyannote enabled, Step 05–07; re-run Step 03b once Step 03 is done so the synthetic half fills in |
| Aug 31–Sep 6 | Step 08, `leave_system_out` sensitivity, freeze figures, draft §1–§4 |
| Sep 7–13 | Human gold returns (`--score-human-gold`); complete draft; internal review |
| Sep 14–16 | Compress to 4 pages, references, submit early |

Confirm the ICASSP 2027 CFP's anonymity and arXiv policy before any preprint.

---

## 14. Layout

```
colab_indicvad.ipynb     Colab GPU notebook (start here)
config.yaml              all hyperparameters + credentials + storage
requirements.txt
main.py                  orchestrator (--steps, --overwrite, --dry-run)
selftest.py              offline known-answer tests; run this first
vadlib/
  config.py              config loading, run-directory layout
  audio.py               I/O, resampling, active-speech SNR, RIR, babble
  corrupt.py             deterministic on-the-fly corruption
  decode.py              posterior -> segments (threshold, durations, hangover)
  metrics.py             DetER, DCF, AUC, EER, boundary F1, segment ratio
  systems.py             VAD wrappers (lazy imports, device routing, batching)
  agreement.py           Fleiss / Cohen kappa for the reference committee
  notify.py              ntfy.sh phone notifications (best-effort)
steps/step01..step11     one module per pipeline stage
  step03b_human_gold.py  standalone annotation pack (run any time, see §3)
  step04b_ssl_probe.py   optional GPU: LOLO-trained SSL probe (see §8)
run/                     created on first run; all outputs land here
```

**Design note.** Step 04 caches frame *posteriors*, not decisions. That single
choice is what makes the 560-setting hyperparameter sweep in Step 07 cost
seconds instead of re-running inference thousands of times — and it is the
mechanism by which apparent "language dependence" can be shown to live in
post-processing rather than in the acoustic model.

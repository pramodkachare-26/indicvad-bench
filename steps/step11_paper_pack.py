"""Step 11 -- assemble everything into 06_paper_pack/ for upload.

Emits a small, self-describing bundle: the CSVs a co-author (or an assistant)
needs to draft the manuscript, plus LaTeX-ready tables, two figures, and a
SUMMARY.md that states the headline numbers in plain language.

Upload the whole 06_paper_pack folder.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CORE = [
    "frame_metrics.csv", "committee_agreement.csv", "ssl_probe_meta.csv",
    "variance_components.csv", "language_effects.csv", "family_effects_secondary.csv",
    "permutation_test.csv", "snr_curves.csv", "hparam_tuning.csv",
    "hparam_global_settings.csv", "phono_features.csv",
    "phono_regression.csv", "lmem_coefficients.csv",
    "human_gold_summary.csv", "human_gold_interannotator.csv", "ssl_layer_probe.csv",
    "transfer_curves.csv", "transfer_typology.csv",
]


def _latex(df, caption, label, floatfmt="%.3f"):
    body = df.to_latex(index=False, float_format=lambda v: floatfmt % v,
                       escape=False, na_rep="--")
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n{body}\\end{{table}}\n")


LANG_NAMES = {
    "as_in": "Assamese", "bn_in": "Bengali", "gu_in": "Gujarati",
    "hi_in": "Hindi", "kn_in": "Kannada", "ml_in": "Malayalam",
    "mr_in": "Marathi", "ne_np": "Nepali", "or_in": "Odia",
    "pa_in": "Punjabi", "ta_in": "Tamil", "te_in": "Telugu", "ur_pk": "Urdu",
}
# 13 series need marker+linestyle redundancy on top of color -- default
# palette repeats after 10 colors and is unreadable in greyscale print.
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8", "p"]
_STYLES = ["-", "--", "-.", ":"]


def _fig_snr(res, paper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = res / "snr_curves.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df.snr_db < 99]
    systems = sorted(df.system.unique())
    langs = sorted(df.lang.unique())
    fig, axes = plt.subplots(1, len(systems), figsize=(3.2 * len(systems), 3.0),
                             sharey=True, squeeze=False)
    for ax, s in zip(axes[0], systems):
        g = df[df.system == s]
        for i, lang in enumerate(langs):
            gg = g[g.lang == lang].sort_values("snr_db")
            if gg.empty:
                continue
            ax.plot(gg.snr_db, gg.deter, marker=_MARKERS[i % len(_MARKERS)],
                     ms=3, lw=1, ls=_STYLES[i % len(_STYLES)],
                     label=LANG_NAMES.get(lang, lang))
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=.3)
    axes[0][0].set_ylabel("DetER")
    axes[0][-1].legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(paper / "fig_snr_by_language.pdf")
    fig.savefig(paper / "fig_snr_by_language.png", dpi=200)
    plt.close(fig)


def _fig_variance(res, paper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = res / "variance_components.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df.factor != "residual"]
    piv = df.pivot_table(index="system", columns="factor", values="omega2")
    fig, ax = plt.subplots(figsize=(6, 3))
    piv.plot(kind="barh", stacked=True, ax=ax)
    ax.set_xlabel(r"$\omega^2$ (share of explained variance)")
    ax.legend(fontsize=7, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(paper / "fig_variance_components.pdf")
    fig.savefig(paper / "fig_variance_components.png", dpi=200)
    plt.close(fig)


def _fig_transfer(res, paper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = res / "transfer_curves.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    zs = df[df.budget_min == 0].pivot_table(index="target", columns="protocol",
                                            values="deter")
    for proto, style in (("lolo+adapt", "-o"), ("lofo+adapt", "--s")):
        g = df[df.protocol == proto]
        if g.empty:
            continue
        m = g.groupby("budget_min")["deter"].mean().sort_index()
        ax.plot(m.index, m.values, style, ms=4, lw=1.4, label=proto)
    for col, ls, lab in (("matched", ":", "matched (upper bound)"),
                         ("lolo", "-.", "zero-shot LOLO")):
        if col in zs.columns:
            ax.axhline(zs[col].mean(), ls=ls, lw=1, color="grey", label=lab)
    ax.set_xlabel("minutes of target-language adaptation audio")
    ax.set_ylabel("DetER")
    ax.set_xscale("symlog", linthresh=1)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(paper / "fig_transfer_curves.pdf")
    fig.savefig(paper / "fig_transfer_curves.png", dpi=200)
    plt.close(fig)


def run(cfg):
    res = Path(cfg["paths"]["results"])
    paper = Path(cfg["paths"]["paper"])
    paper.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in CORE:
        src = res / name
        if src.exists():
            shutil.copy(src, paper / name)
            copied.append(name)

    tex = []
    vc_p = paper / "variance_components.csv"
    headline = {}
    if vc_p.exists():
        vc = pd.read_csv(vc_p)
        t = vc[vc.factor != "residual"].pivot_table(
            index="factor", columns="system", values="omega2").reset_index()
        tex.append(_latex(t, "Variance components ($\\omega^2$) of VAD frame "
                             "error by factor.", "tab:variance"))
        m = vc[vc.factor != "residual"].groupby("factor")["omega2"].mean()
        if len(m):
            headline["omega2_by_factor"] = m.round(4).to_dict()
            headline["top_factor"] = str(m.idxmax())
            headline["language_share"] = float(m.get("lang", np.nan))

    hp_p = paper / "hparam_tuning.csv"
    if hp_p.exists():
        hp = pd.read_csv(hp_p)
        t = hp.groupby("system")[["deter_global", "deter_perlang",
                                  "deter_oracle", "gap_closed_frac",
                                  "rel_improvement"]].mean().reset_index()
        tex.append(_latex(t, "Per-language decoding retuning: global vs "
                             "per-language vs oracle DetER.", "tab:hparam"))
        headline["mean_rel_improvement_retuning"] = float(hp.rel_improvement.mean())
        headline["mean_gap_closed"] = float(hp.gap_closed_frac.mean(skipna=True))

    le_p = paper / "language_effects.csv"
    if le_p.exists():
        le = pd.read_csv(le_p)
        t = le.pivot_table(index=["lang", "family"], columns="system",
                           values="mean").reset_index()
        tex.append(_latex(t, "Mean DetER by language (noisy conditions).",
                          "tab:langmeans"))
        spread = le.groupby("system")["mean"].agg(lambda v: v.max() - v.min())
        headline["max_language_spread_deter"] = float(spread.mean())

    tc_p = paper / "transfer_curves.csv"
    if tc_p.exists():
        tc = pd.read_csv(tc_p)
        zs = tc[tc.budget_min == 0].pivot_table(index="target",
                                                columns="protocol",
                                                values="deter").reset_index()
        tex.append(_latex(zs, "Zero-shot DetER by transfer protocol (RQ3).",
                          "tab:transfer"))
        z = zs.set_index("target")
        if {"lolo", "matched"}.issubset(z.columns):
            headline["unseen_language_penalty"] = float(
                (z["lolo"] - z["matched"]).mean())
        if {"lofo", "lolo"}.issubset(z.columns):
            headline["extra_cost_crossing_family"] = float(
                (z["lofo"] - z["lolo"]).mean())

    ia_p = paper / "human_gold_interannotator.csv"
    if ia_p.exists():
        ia = pd.read_csv(ia_p)
        for src, g in ia.groupby("source"):
            headline[f"human_gold_kappa_{src}"] = float(g.cohen_kappa.mean())
            headline[f"human_gold_onset_mad_s_{src}"] = float(g.onset_mad_s.median())
            headline[f"human_gold_offset_mad_s_{src}"] = float(g.offset_mad_s.median())

    (paper / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
    _fig_snr(res, paper)
    _fig_variance(res, paper)
    _fig_transfer(res, paper)

    lines = [
        "# IndicVAD-Bench -- results summary", "",
        "Generated by `main.py --steps 11`. Upload this whole folder.", "",
        "## Files", "",
    ]
    lines += [f"- `{c}`" for c in copied]
    lines += ["- `tables.tex` (LaTeX-ready)",
              "- `fig_snr_by_language.pdf`, `fig_variance_components.pdf`", ""]
    lines += ["## Headline numbers", ""]
    if headline:
        for k, v in headline.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- (no analysis CSVs found -- run steps 05-08 first)")
    lines += ["", "## How to read this", "",
              "1. `variance_components.csv` answers RQ1. If `lang` has a small",
              "   omega^2 relative to `snr_db`/`reverb`, the honest conclusion is",
              "   that VAD error in Indian languages is channel-driven, not",
              "   language-driven. Report that plainly; it is the paper's claim.",
              "2. `permutation_test.csv` guards against reading noise as a",
              "   language effect. Check the p-value before claiming either way.",
              "3. `hparam_tuning.csv` answers RQ3-lite and is the positive",
              "   result: how much of the apparent language gap is absorbed by",
              "   retuning three decoding scalars, with no retraining.",
              "4. `phono_regression.csv` answers RQ2. Rows with",
              "   `controls_language=True` are the meaningful ones: a density",
              "   effect that survives language dummies is a real phonological",
              "   effect, not a relabelled language effect.",
              "5. `transfer_curves.csv` answers full RQ3. The two numbers that",
              "   matter: the unseen-language penalty (lolo minus matched) and",
              "   the extra cost of crossing a family boundary (lofo minus",
              "   lolo). The adaptation curve says how many minutes of target",
              "   audio close the gap.",
              "6. `transfer_typology.csv` has n=8 languages. Report the",
              "   correlation descriptively; it is NOT powered for inference.",
              "7. `ssl_layer_probe.csv` justifies the encoder layer used. If",
              "   the best layer is not the last, say so -- most papers take",
              "   the last hidden state without checking.",
              "8. `human_gold_summary.csv` is your external-validity anchor. If",
              "   it is missing, say so in the limitations rather than implying",
              "   the synthetic benchmark was validated against human labels.",
              "9. `human_gold_interannotator.csv` has Cohen's kappa and onset/",
              "   offset boundary MAD between the two annotators, split by",
              "   `source`. Report synthetic and real subsets SEPARATELY --",
              "   they answer different questions and must not be pooled.",
              ""]
    (paper / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    # On Colab the scratch disk dies with the VM. Everything precious is
    # already on Drive by construction, but state that plainly so nobody
    # goes looking for posteriors after a disconnect.
    if cfg.get("storage_mode") == "colab":
        print(f"\n[step11] Drive (persists): {cfg['paths']['drive_root']}")
        print(f"[step11] scratch (LOST on disconnect): "
              f"{cfg['paths']['scratch_root']}")
        print("[step11] results, paper pack and logs are on Drive. "
              "Posteriors are not -- rerun step 04 if you need them again.")

    print(f"\n[step11] paper pack ready: {paper}")
    for f in sorted(paper.iterdir()):
        print(f"   {f.name:38s} {f.stat().st_size/1024:8.1f} KB")
    return paper

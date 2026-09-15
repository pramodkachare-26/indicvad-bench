"""Step 06 -- the headline analysis: how much error variance is LANGUAGE?

Produces
--------
variance_components.csv : omega^2 / partial eta^2 per factor, per system.
                          THIS TABLE IS THE PAPER'S CENTRAL CLAIM.
language_effects.csv    : per-language marginal means with bootstrap 95% CIs.
lmem_coefficients.csv    : mixed-effects coefficients (random intercept per
                          session) if statsmodels is available.
permutation_test.csv    : p-value for the language factor under label
                          permutation -- guards against reading noise as signal.

A null result here is publishable IF paired with Step 07. Say so plainly in
the paper rather than burying it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6


# --------------------------------------------------------------------------- #
def _logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def _design(df, factors):
    """Dummy-coded design matrix plus a column->factor map."""
    blocks, owner = [np.ones((len(df), 1))], ["intercept"]
    for f in factors:
        v = df[f]
        if pd.api.types.is_numeric_dtype(v) and v.nunique() > 6:
            blocks.append(v.to_numpy(dtype=np.float64)[:, None])
            owner.append(f)
        else:
            d = pd.get_dummies(v.astype(str), prefix=f, drop_first=True)
            if d.shape[1] == 0:
                continue
            blocks.append(d.to_numpy(dtype=np.float64))
            owner.extend([f] * d.shape[1])
    return np.hstack(blocks), np.array(owner)


def _ssr(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r), beta


def variance_components(df, factors, ycol):
    """Partial eta^2 and omega^2 by leave-one-factor-out refitting."""
    y = df[ycol].to_numpy(dtype=np.float64)
    X, owner = _design(df, factors)
    ss_full, _ = _ssr(X, y)
    n, k = X.shape
    df_res = max(n - k, 1)
    ms_res = ss_full / df_res
    ss_tot = float(((y - y.mean()) ** 2).sum())

    out = []
    for f in factors:
        keep = owner != f
        ss_red, _ = _ssr(X[:, keep], y)
        d_ss = max(ss_red - ss_full, 0.0)
        d_df = max(int((~keep).sum()), 1)
        out.append({
            "factor": f, "df": d_df, "ss": d_ss,
            "partial_eta2": d_ss / (d_ss + ss_full) if (d_ss + ss_full) > 0 else np.nan,
            "omega2": max((d_ss - d_df * ms_res) / (ss_tot + ms_res), 0.0)
            if ss_tot > 0 else np.nan,
            "F": (d_ss / d_df) / ms_res if ms_res > 0 else np.nan,
        })
    out.append({"factor": "residual", "df": df_res, "ss": ss_full,
                "partial_eta2": np.nan,
                "omega2": ss_full / (ss_tot + ms_res) if ss_tot > 0 else np.nan,
                "F": np.nan})
    return pd.DataFrame(out)


def permutation_p(df, factors, ycol, target="lang", n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    y = df[ycol].to_numpy(dtype=np.float64)
    X, owner = _design(df, factors)
    ss_full, _ = _ssr(X, y)
    ss_red, _ = _ssr(X[:, owner != target], y)
    obs = ss_red - ss_full

    work = df.copy()
    ge = 0
    for _ in range(int(n_perm)):
        work[target] = rng.permutation(work[target].to_numpy())
        Xp, ownp = _design(work, factors)
        sf, _ = _ssr(Xp, y)
        sr, _ = _ssr(Xp[:, ownp != target], y)
        if (sr - sf) >= obs:
            ge += 1
    return (ge + 1) / (n_perm + 1), obs


def bootstrap_means(df, group_cols, value_col, unit_col="session_id",
                    n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for keys, g in df.groupby(group_cols):
        units = g[unit_col].unique()
        by_unit = {u: g.loc[g[unit_col] == u, value_col].to_numpy() for u in units}
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.choice(len(units), size=len(units), replace=True)
            boots[b] = np.concatenate([by_unit[units[i]] for i in pick]).mean()
        rec = dict(zip(group_cols if isinstance(group_cols, list) else [group_cols],
                       keys if isinstance(keys, tuple) else (keys,)))
        rec.update({
            "mean": float(g[value_col].mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "n": int(len(g)),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def _try_mixedlm(df, ycol):
    try:
        import statsmodels.formula.api as smf
    except Exception:
        print("[step06] statsmodels not installed; skipping mixed-effects model")
        return None
    try:
        m = smf.mixedlm(f"{ycol} ~ C(lang) + snr_db + C(noise) + C(reverb)",
                        df, groups=df["session_idx"]).fit(reml=True)
        return pd.DataFrame({"term": m.params.index, "coef": m.params.values,
                             "se": m.bse.reindex(m.params.index).values,
                             "pvalue": m.pvalues.reindex(m.params.index).values})
    except Exception as e:
        print(f"[step06] mixedlm failed ({e}); continuing without it")
        return None


# --------------------------------------------------------------------------- #
def run(cfg):
    res = Path(cfg["paths"]["results"])
    df = pd.read_csv(res / "frame_metrics.csv")
    st = cfg["stats"]

    total = df["n_speech_s"] + df["n_nonspeech_s"]
    df["err_rate"] = (df["miss_s"] + df["fa_s"]) / total.clip(lower=EPS)
    df["y"] = _logit(df["err_rate"])
    noisy = df[df["snr_db"] < 99].copy()

    factors = ["lang", "snr_db", "noise", "reverb"]

    vc_all, perm_rows = [], []
    for sysname, g in noisy.groupby("system"):
        if g["lang"].nunique() < 2:
            continue
        vc = variance_components(g, factors, "y")
        vc.insert(0, "system", sysname)
        vc_all.append(vc)

        p, obs = permutation_p(g, factors, "y", "lang",
                               n_perm=int(st["n_permutation"]),
                               seed=int(cfg["project"]["seed"]))
        perm_rows.append({"system": sysname, "factor": "lang",
                          "observed_delta_ss": obs, "p_value": p,
                          "n_perm": int(st["n_permutation"])})
        print(f"[step06] {sysname}: language permutation p = {p:.4f}")

    vc_df = pd.concat(vc_all, ignore_index=True)
    vc_df.to_csv(res / "variance_components.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(res / "permutation_test.csv", index=False)

    lang_eff = bootstrap_means(noisy, ["system", "lang", "family"], "deter",
                               n_boot=int(st["n_bootstrap"]),
                               seed=int(cfg["project"]["seed"]))
    lang_eff.to_csv(res / "language_effects.csv", index=False)

    # --- family: descriptive covariate, NOT a headline group contrast -------
    # `family` is not fit as a factor above; `lang` already absorbs it. With
    # an imbalanced split (e.g. 9 Indo-Aryan vs 4 Dravidian in the default
    # config) a two-group family test would be underpowered and easy to
    # over-read. Report it only as a secondary, explicitly-labeled aggregate.
    fam_counts = noisy.drop_duplicates("lang").groupby("family").size()
    if len(fam_counts) > 1 and (fam_counts.max() / fam_counts.min()) >= 1.5:
        print(f"[step06] NOTE: language families are imbalanced "
              f"({fam_counts.to_dict()}). Reporting family only as a "
              f"descriptive secondary aggregate, not a headline contrast -- "
              f"RQ1's claim is the per-language variance decomposition above.")
    family_eff = bootstrap_means(noisy, ["system", "family"], "deter",
                                 n_boot=int(st["n_bootstrap"]),
                                 seed=int(cfg["project"]["seed"]))
    family_eff.insert(1, "role", "secondary_descriptive_not_headline")
    family_eff.to_csv(res / "family_effects_secondary.csv", index=False)

    lmm = _try_mixedlm(noisy, "y")
    if lmm is not None:
        lmm.to_csv(res / "lmem_coefficients.csv", index=False)

    snr_curve = (noisy.groupby(["system", "lang", "snr_db"])
                 [["deter", "p_miss", "p_fa", "auc"]].mean().reset_index())
    snr_curve.to_csv(res / "snr_curves.csv", index=False)

    print("\n[step06] HEADLINE -- omega^2 by factor (share of variance):")
    head = vc_df.pivot_table(index="factor", columns="system", values="omega2")
    print(head.round(4).to_string())
    print(f"\n[step06] wrote 5 CSVs to {res}")
    return res / "variance_components.csv"

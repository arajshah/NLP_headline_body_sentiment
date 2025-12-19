from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class GapStats:
    n: int
    mean_gap: float
    median_gap: float
    std_gap: float
    pct_neg_gap: float
    pct_pos_gap: float
    ttest_p: float
    wilcoxon_p: float
    cohens_d: float
    ci95_low: float
    ci95_high: float


def bootstrap_ci(x: np.ndarray, *, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compute_gap(df: pd.DataFrame, *, head_col: str = "sent_head", body_col: str = "sent_body") -> pd.DataFrame:
    out = df.copy()
    out[head_col] = pd.to_numeric(out[head_col], errors="coerce")
    out[body_col] = pd.to_numeric(out[body_col], errors="coerce")
    out["sent_gap"] = out[head_col] - out[body_col]
    return out


def summarize_gap(df: pd.DataFrame) -> GapStats:
    x = pd.to_numeric(df["sent_gap"], errors="coerce").to_numpy()
    x = x[~np.isnan(x)]
    n = int(len(x))
    if n == 0:
        return GapStats(
            n=0,
            mean_gap=np.nan,
            median_gap=np.nan,
            std_gap=np.nan,
            pct_neg_gap=np.nan,
            pct_pos_gap=np.nan,
            ttest_p=np.nan,
            wilcoxon_p=np.nan,
            cohens_d=np.nan,
            ci95_low=np.nan,
            ci95_high=np.nan,
        )

    mean_gap = float(np.mean(x))
    median_gap = float(np.median(x))
    std_gap = float(np.std(x, ddof=1)) if n > 1 else 0.0
    pct_neg = float(np.mean(x < 0) * 100.0)
    pct_pos = float(np.mean(x > 0) * 100.0)

    # tests vs 0
    t_p = float(stats.ttest_1samp(x, 0.0).pvalue) if n > 1 else np.nan
    # Wilcoxon requires non-zero differences; scipy will error if all zero
    try:
        w_p = float(stats.wilcoxon(x).pvalue)
    except Exception:
        w_p = np.nan

    # Cohen's d for one-sample (mean / std)
    d = float(mean_gap / std_gap) if std_gap and not np.isnan(std_gap) else np.nan

    ci_low, ci_high = bootstrap_ci(x)

    return GapStats(
        n=n,
        mean_gap=mean_gap,
        median_gap=median_gap,
        std_gap=std_gap,
        pct_neg_gap=pct_neg,
        pct_pos_gap=pct_pos,
        ttest_p=t_p,
        wilcoxon_p=w_p,
        cohens_d=d,
        ci95_low=ci_low,
        ci95_high=ci_high,
    )


def group_gap(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str],
    min_n: int = 30,
) -> pd.DataFrame:
    """
    Group-level gap summary with sample size filtering.
    """
    if "sent_gap" not in df.columns:
        raise ValueError("sent_gap is required (run compute_gap first).")

    g = (
        df.dropna(subset=["sent_gap"])
        .groupby(list(group_cols), dropna=False)
        .agg(
            n=("sent_gap", "size"),
            mean_gap=("sent_gap", "mean"),
            median_gap=("sent_gap", "median"),
            std_gap=("sent_gap", "std"),
        )
        .reset_index()
    )
    return g[g["n"] >= min_n].sort_values("n", ascending=False).reset_index(drop=True)



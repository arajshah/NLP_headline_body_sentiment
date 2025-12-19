from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VaderConfig:
    """
    VADER outputs compound scores in [-1, 1]. This provides a fast,
    offline robustness baseline for transformer sentiment.
    """

    pass


def score_vader(
    df: pd.DataFrame,
    *,
    headline_col: str = "headline_clean",
    body_col: str = "body_clean",
    id_col: str = "id",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    out = df.copy()
    for col in [id_col, headline_col, body_col]:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")

    cache = None
    if cache_path is not None and cache_path.exists():
        cache = pd.read_csv(cache_path).drop_duplicates(subset=[id_col])
        out = out.merge(cache, on=id_col, how="left", suffixes=("", "_cache"))
    else:
        out["sent_head_vader"] = np.nan
        out["sent_body_vader"] = np.nan

    todo = out[out["sent_head_vader"].isna() | out["sent_body_vader"].isna()].copy()
    if len(todo) == 0:
        return out

    analyzer = SentimentIntensityAnalyzer()

    def compound(x: str) -> float:
        if not isinstance(x, str) or x.strip() == "":
            return np.nan
        return float(analyzer.polarity_scores(x)["compound"])

    todo["sent_head_vader"] = todo[headline_col].apply(compound)
    todo["sent_body_vader"] = todo[body_col].apply(compound)

    computed = todo[[id_col, "sent_head_vader", "sent_body_vader"]].copy()
    out = out.merge(computed, on=id_col, how="left", suffixes=("", "_new"))
    out["sent_head_vader"] = out["sent_head_vader"].fillna(out["sent_head_vader_new"])
    out["sent_body_vader"] = out["sent_body_vader"].fillna(out["sent_body_vader_new"])
    out = out.drop(columns=[c for c in ["sent_head_vader_new", "sent_body_vader_new"] if c in out.columns])

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out[[id_col, "sent_head_vader", "sent_body_vader"]].to_csv(cache_path, index=False)

    return out



from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BiasMergeConfig:
    bias_csv: Path
    bias_domain_col: str = "domain"
    bias_score_col: str = "bias_rating"
    bias_url_col: str = "url"  # if provided table is URL-based


def base_domain(url: str) -> str | float:
    """
    Return 'cnn.com' from 'https://edition.cnn.com/foo' (or np.nan).
    Uses tldextract to handle subdomains/TLDs.
    """
    if not isinstance(url, str) or url.strip() == "":
        return np.nan
    u = url.strip().lower()

    import tldextract

    ext = tldextract.extract(u)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return np.nan


def attach_domain(df: pd.DataFrame, *, url_col: str = "url") -> pd.DataFrame:
    out = df.copy()
    out[url_col] = out[url_col].astype(str).str.strip()
    out["domain"] = out[url_col].apply(base_domain)
    return out


def load_bias_table(cfg: BiasMergeConfig) -> pd.DataFrame:
    raw = pd.read_csv(cfg.bias_csv)
    if cfg.bias_domain_col in raw.columns and cfg.bias_score_col in raw.columns:
        bias = raw[[cfg.bias_domain_col, cfg.bias_score_col]].copy()
        bias = bias.dropna(subset=[cfg.bias_domain_col]).drop_duplicates(cfg.bias_domain_col)
        bias = bias.rename(columns={cfg.bias_domain_col: "domain"})
        return bias

    # else: attempt URL-based table
    if cfg.bias_url_col not in raw.columns or cfg.bias_score_col not in raw.columns:
        raise ValueError(
            f"Bias table must contain either ({cfg.bias_domain_col},{cfg.bias_score_col}) "
            f"or ({cfg.bias_url_col},{cfg.bias_score_col}). Found: {list(raw.columns)}"
        )
    tmp = raw.rename(columns={cfg.bias_url_col: "_bias_url"})[["_bias_url", cfg.bias_score_col]].copy()
    tmp["domain"] = tmp["_bias_url"].apply(base_domain)
    bias = tmp.dropna(subset=["domain"]).drop_duplicates("domain")[["domain", cfg.bias_score_col]]
    return bias


def merge_bias(df: pd.DataFrame, *, bias_df: pd.DataFrame) -> pd.DataFrame:
    if "domain" not in df.columns:
        raise ValueError("df must contain 'domain' (run attach_domain first).")
    if "domain" not in bias_df.columns:
        raise ValueError("bias_df must contain 'domain'.")
    out = df.merge(bias_df, on="domain", how="left")
    return out



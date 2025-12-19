from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .cleaning import light_clean


BodyMode = Literal["summary", "fulltext"]


@dataclass(frozen=True)
class CanonicalColumns:
    id: str = "id"
    url: str = "url"
    headline: str = "headline_text"
    body: str = "body_text"
    body_source: str = "body_source"  # summary | scraped
    q2_focus: str = "Q2 Focus"
    q3_theme1: str = "Q3 Theme1"
    q3_theme2: str = "Q3 Theme2"
    headline_clean: str = "headline_clean"
    body_clean: str = "body_clean"


def _require_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {where}: {missing}. Present: {list(df.columns)}")


def load_gvfc_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_cols(
        df,
        [
            "id",
            "article_url",
            "headline",
            "presum_summary_of_full_article_text",
            "Q2 Focus",
            "Q3 Theme1",
            "Q3 Theme2",
        ],
        where=str(path),
    )
    return df


def build_canonical_dataset(
    gvfc_master_csv: Path,
    *,
    body_mode: BodyMode = "summary",
    full_text_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build the canonical dataset used across the repo.

    body_mode='summary':
      body_text := presum_summary_of_full_article_text

    body_mode='fulltext':
      body_text := scraped full text from full_text_dir/{id}.txt,
      with fallback to the summary if missing.
    """
    cols = CanonicalColumns()
    master = load_gvfc_master(gvfc_master_csv)

    df = master[
        [
            "id",
            "article_url",
            "headline",
            "presum_summary_of_full_article_text",
            cols.q2_focus,
            cols.q3_theme1,
            cols.q3_theme2,
        ]
    ].copy()

    df = df.rename(
        columns={
            "article_url": cols.url,
            "headline": cols.headline,
            "presum_summary_of_full_article_text": "_summary_body",
        }
    )

    # Basic string hygiene (important for downstream domain extraction)
    # Keep NaNs as NaNs until after we drop missing values.
    df[cols.url] = df[cols.url].astype(str).str.strip()

    if body_mode == "summary":
        df[cols.body] = df["_summary_body"]
        df[cols.body_source] = "summary"
    else:
        if full_text_dir is None:
            raise ValueError("full_text_dir is required when body_mode='fulltext'")
        # Load scraped texts, if present
        texts: dict[int, str] = {}
        if full_text_dir.exists():
            for p in full_text_dir.glob("*.txt"):
                try:
                    art_id = int(p.stem)
                except ValueError:
                    continue
                try:
                    texts[art_id] = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

        df[cols.body] = df["id"].map(texts)
        df["_body_source_scraped"] = df[cols.body].notna()
        df[cols.body] = df[cols.body].fillna(df["_summary_body"])
        df[cols.body_source] = df["_body_source_scraped"].map(lambda x: "scraped" if x else "summary")

    # Drop rows missing headline/body after construction
    df = df.dropna(subset=[cols.headline, cols.body]).copy()

    # Cast after we have removed missing values.
    df[cols.headline] = df[cols.headline].astype(str)
    df[cols.body] = df[cols.body].astype(str)

    # Clean text fields
    df[cols.headline_clean] = df[cols.headline].apply(light_clean)
    df[cols.body_clean] = df[cols.body].apply(light_clean)

    # Stable ordering
    df = df.sort_values(cols.id).reset_index(drop=True)
    df = df.drop(columns=[c for c in ["_summary_body", "_body_source_scraped"] if c in df.columns])

    return df



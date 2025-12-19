from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SentimentConfig:
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokens_per_chunk: int = 450
    max_length: int = 512
    batch_size: int = 32


LABEL2SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def weighted_score(result: list[dict]) -> float:
    """
    Convert a list of {'label','score'} dicts (one item per class)
    into a single signed sentiment score in [-1, 1].
    """
    return float(sum(d["score"] * LABEL2SCORE[d["label"].lower()] for d in result))


def chunk_by_tokens(text: str, tokenizer, tokens_per_chunk: int) -> Iterable[str]:
    """
    Split long text into chunks by token count (no sentence splitting).
    Uses tokenizer.encode/decode to avoid repeated tokenization work.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    for i in range(0, len(ids), tokens_per_chunk):
        chunk_ids = ids[i : i + tokens_per_chunk]
        yield tokenizer.decode(chunk_ids, skip_special_tokens=True)


def load_sentiment_pipeline(cfg: SentimentConfig):
    """
    Lazily import transformers/torch so the rest of the project remains usable
    even if ML deps aren't installed.
    """
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        top_k=None,  # modern replacement for return_all_scores=True
    )
    return pipe, tokenizer


def score_sentiment(
    df: pd.DataFrame,
    *,
    headline_col: str = "headline_clean",
    body_col: str = "body_clean",
    id_col: str = "id",
    cache_path: Path | None = None,
    cfg: SentimentConfig = SentimentConfig(),
) -> pd.DataFrame:
    """
    Add sentiment scores for headline and body.
    If cache_path is provided, cache per-id results to avoid recompute.
    """
    out = df.copy()
    for col in [id_col, headline_col, body_col]:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")

    # Simple cache: CSV with columns [id, sent_head, sent_body]
    cache = None
    if cache_path is not None and cache_path.exists():
        cache = pd.read_csv(cache_path)
        cache = cache.drop_duplicates(subset=[id_col])

    if cache is not None:
        out = out.merge(cache, on=id_col, how="left", suffixes=("", "_cache"))
    else:
        out["sent_head"] = np.nan
        out["sent_body"] = np.nan

    todo = out[out["sent_head"].isna() | out["sent_body"].isna()].copy()
    if len(todo) == 0:
        return out

    pipe, tokenizer = load_sentiment_pipeline(cfg)

    def score_headline(x: str) -> float:
        res = pipe(x)
        # pipeline returns list[list[dict]] because top_k=None
        return weighted_score(res[0])

    def score_body(x: str) -> float:
        if not isinstance(x, str) or x.strip() == "":
            return np.nan
        chunks = list(chunk_by_tokens(x, tokenizer, cfg.tokens_per_chunk))
        res = pipe(chunks)  # list[list[dict]]
        scores = [weighted_score(r) for r in res]
        return float(np.mean(scores)) if scores else np.nan

    # Compute on the missing subset
    todo["sent_head"] = todo[headline_col].apply(score_headline)
    todo["sent_body"] = todo[body_col].apply(score_body)

    # Merge back, preserving any cached rows.
    computed = todo[[id_col, "sent_head", "sent_body"]].copy()
    out = out.merge(computed, on=id_col, how="left", suffixes=("", "_new"))
    out["sent_head"] = out["sent_head"].fillna(out["sent_head_new"])
    out["sent_body"] = out["sent_body"].fillna(out["sent_body_new"])
    out = out.drop(columns=[c for c in ["sent_head_new", "sent_body_new"] if c in out.columns])

    # Persist cache
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out[[id_col, "sent_head", "sent_body"]].to_csv(cache_path, index=False)

    return out



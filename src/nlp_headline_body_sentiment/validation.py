from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_canonical(df: pd.DataFrame, *, required_cols: Iterable[str]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    required_cols = list(required_cols)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    if df["id"].isna().any():
        errors.append("Found NaN ids.")
    if df["id"].duplicated().any():
        dup_n = int(df["id"].duplicated().sum())
        errors.append(f"Found duplicate ids (n={dup_n}).")

    if "url" in df.columns:
        empty_url = (df["url"].astype(str).str.strip() == "").sum()
        if empty_url:
            errors.append(f"Found empty urls (n={int(empty_url)}).")
        # common hygiene issue observed in your notebook output
        leading_ws = df["url"].astype(str).str.match(r"^\s+").sum()
        if leading_ws:
            warnings.append(f"Found urls with leading whitespace/newlines (n={int(leading_ws)}).")

    # Annotation code sanity checks (non-fatal; dataset may use special codes like 99)
    for col in ["Q2 Focus", "Q3 Theme1", "Q3 Theme2"]:
        if col in df.columns:
            bad = pd.to_numeric(df[col], errors="coerce").isna().sum()
            if bad:
                warnings.append(f"{col}: {int(bad)} non-numeric values (may be missing/unknown codes).")

    return ValidationResult(ok=(len(errors) == 0), errors=errors, warnings=warnings)



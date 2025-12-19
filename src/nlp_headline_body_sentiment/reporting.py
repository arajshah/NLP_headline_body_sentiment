from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .analysis import summarize_gap


def write_markdown_report(
    *,
    mode: str,
    df_gap: pd.DataFrame,
    out_path: Path,
    figures_dir: Path,
    extra_tables: dict[str, Path] | None = None,
) -> None:
    stats = summarize_gap(df_gap)
    extra_tables = extra_tables or {}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = []
    md.append(f"## Headline–Body Sentiment Gap Report ({mode})\n")
    md.append("### Summary\n")
    md.append(f"- **n**: {stats.n}\n")
    md.append(f"- **mean gap**: {stats.mean_gap:.4f} (95% CI {stats.ci95_low:.4f} to {stats.ci95_high:.4f})\n")
    md.append(f"- **median gap**: {stats.median_gap:.4f}\n")
    md.append(f"- **std gap**: {stats.std_gap:.4f}\n")
    md.append(f"- **pct gap < 0**: {stats.pct_neg_gap:.1f}%\n")
    md.append(f"- **pct gap > 0**: {stats.pct_pos_gap:.1f}%\n")
    md.append(f"- **t-test vs 0 p**: {stats.ttest_p:.3g}\n")
    md.append(f"- **Wilcoxon vs 0 p**: {stats.wilcoxon_p:.3g}\n")
    md.append(f"- **Cohen’s d (one-sample)**: {stats.cohens_d:.3f}\n")

    md.append("\n### Figures\n")
    for name in ["hist_gap.png", "box_gap.png", "scatter_head_body.png"]:
        p = figures_dir / name
        if p.exists():
            # relative link for GitHub rendering
            rel = p.as_posix()
            md.append(f"- `{name}`: ![]({rel})\n")
        else:
            md.append(f"- `{name}`: (not found)\n")

    if extra_tables:
        md.append("\n### Tables\n")
        for label, path in extra_tables.items():
            md.append(f"- **{label}**: `{path.as_posix()}`\n")

    out_path.write_text("".join(md), encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



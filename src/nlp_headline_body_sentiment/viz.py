from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_gap_figures(df: pd.DataFrame, *, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # Histogram
    plt.figure(figsize=(7, 4))
    sns.histplot(df["sent_gap"], bins=40, kde=True)
    plt.axvline(0, color="black", lw=1)
    plt.xlabel("Headline sentiment – Body sentiment")
    plt.ylabel("Article count")
    plt.title("Distribution of sentiment gaps")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_gap.png", dpi=300)
    plt.close()

    # Box
    plt.figure(figsize=(4, 4))
    sns.boxplot(y=df["sent_gap"])
    plt.ylabel("Headline – Body sentiment")
    plt.title("Sentiment gap (boxplot)")
    plt.tight_layout()
    plt.savefig(out_dir / "box_gap.png", dpi=300)
    plt.close()

    # Scatter
    if "sent_head" in df.columns and "sent_body" in df.columns:
        plt.figure(figsize=(5, 5))
        sns.scatterplot(x="sent_body", y="sent_head", data=df, alpha=0.35, s=18)
        plt.axline((0, -1), slope=1, color="red", ls="--", lw=1)
        plt.xlabel("Body sentiment")
        plt.ylabel("Headline sentiment")
        plt.title("Headline vs body sentiment")
        plt.tight_layout()
        plt.savefig(out_dir / "scatter_head_body.png", dpi=300)
        plt.close()



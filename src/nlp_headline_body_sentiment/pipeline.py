from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console

from .analysis import compare_models, compute_gap, group_gap, summarize_gap
from .bias import BiasMergeConfig, attach_domain, load_bias_table, merge_bias
from .datasets import build_canonical_dataset
from .paths import ProjectPaths, find_repo_root
from .sentiment import SentimentConfig, score_sentiment
from .validation import validate_canonical
from .vader_sentiment import score_vader
from .viz import save_gap_figures
from .reporting import write_json, write_markdown_report


console = Console()


def cmd_run_all(args: argparse.Namespace) -> None:
    paths = ProjectPaths(find_repo_root())
    paths.ensure_dirs()

    gvfc_master = paths.data_dir / "GVFC_extension_multimodal.csv"
    if not gvfc_master.exists():
        raise SystemExit(f"Missing required input: {gvfc_master}")

    # 1) Build canonical dataset
    df = build_canonical_dataset(
        gvfc_master,
        body_mode=args.mode,
        full_text_dir=paths.full_text_dir,
    )
    vr = validate_canonical(
        df,
        required_cols=["id", "url", "headline_text", "body_text", "headline_clean", "body_clean", "body_source"],
    )
    if not vr.ok:
        raise SystemExit("Dataset validation failed:\n- " + "\n- ".join(vr.errors))
    for w in vr.warnings:
        console.print(f"[yellow]⚠[/yellow] {w}")

    out_base = paths.derived_dir / f"articles_{args.mode}.csv"
    df.to_csv(out_base, index=False)
    console.print(f"[green]✓[/green] wrote {out_base} ({len(df):,} rows)")

    # 2) Sentiment
    cfg = SentimentConfig(model_name=args.model)
    cache = paths.derived_dir / f"sentiment_cache_transformer_{args.mode}.csv"
    df2 = score_sentiment(df, cache_path=cache, cfg=cfg)
    if args.vader:
        vcache = paths.derived_dir / f"sentiment_cache_vader_{args.mode}.csv"
        df2 = score_vader(df2, cache_path=vcache)
    out_sent = paths.derived_dir / f"articles_{args.mode}_with_sentiment.csv"
    df2.to_csv(out_sent, index=False)
    console.print(f"[green]✓[/green] wrote {out_sent}")

    # 3) Gap + stats
    df3 = compute_gap(df2)
    out_gap = paths.derived_dir / f"articles_{args.mode}_with_gap.csv"
    df3.to_csv(out_gap, index=False)
    console.print(f"[green]✓[/green] wrote {out_gap}")

    stats_ = summarize_gap(df3)
    out_stats = paths.derived_dir / f"gap_summary_{args.mode}.json"
    write_json(out_stats, stats_.__dict__)
    console.print(f"[green]✓[/green] wrote {out_stats}")

    # 4) Subgroup summaries (annotations)
    out_group = paths.derived_dir / f"gap_by_focus_{args.mode}.csv"
    group_gap(df3, group_cols=["Q2 Focus"], min_n=args.min_group_n).to_csv(out_group, index=False)
    console.print(f"[green]✓[/green] wrote {out_group}")

    out_group2 = paths.derived_dir / f"gap_by_theme1_{args.mode}.csv"
    group_gap(df3, group_cols=["Q3 Theme1"], min_n=args.min_group_n).to_csv(out_group2, index=False)
    console.print(f"[green]✓[/green] wrote {out_group2}")

    out_group3 = paths.derived_dir / f"gap_by_theme2_{args.mode}.csv"
    group_gap(df3, group_cols=["Q3 Theme2"], min_n=args.min_group_n).to_csv(out_group3, index=False)
    console.print(f"[green]✓[/green] wrote {out_group3}")

    if args.vader:
        comp = compare_models(df3)
        out_comp = paths.derived_dir / f"model_comparison_{args.mode}.csv"
        comp.to_csv(out_comp, index=False)
        console.print(f"[green]✓[/green] wrote {out_comp}")

    # 5) Figures
    fig_dir = paths.figures_dir / args.mode
    save_gap_figures(df3, out_dir=fig_dir)
    console.print(f"[green]✓[/green] wrote figures under {fig_dir}")

    # 5b) Report (markdown)
    report_path = paths.reports_dir / f"report_{args.mode}.md"
    write_markdown_report(
        mode=args.mode,
        df_gap=df3,
        out_path=report_path,
        figures_dir=fig_dir,
        extra_tables={
            "Gap by focus": out_group,
            "Gap by theme1": out_group2,
            "Gap by theme2": out_group3,
        },
    )
    console.print(f"[green]✓[/green] wrote {report_path}")

    # 6) Optional bias merge
    if args.bias_csv:
        df_b = attach_domain(df3, url_col="url")
        bias_df = load_bias_table(BiasMergeConfig(bias_csv=Path(args.bias_csv)))
        df_b2 = merge_bias(df_b, bias_df=bias_df)
        out_bias = paths.derived_dir / f"articles_{args.mode}_with_bias.csv"
        df_b2.to_csv(out_bias, index=False)
        console.print(f"[green]✓[/green] wrote {out_bias}")


def cmd_build(args: argparse.Namespace) -> None:
    paths = ProjectPaths(find_repo_root())
    paths.ensure_dirs()
    gvfc_master = paths.data_dir / "GVFC_extension_multimodal.csv"
    df = build_canonical_dataset(gvfc_master, body_mode=args.mode, full_text_dir=paths.full_text_dir)
    out_base = paths.derived_dir / f"articles_{args.mode}.csv"
    df.to_csv(out_base, index=False)
    console.print(f"[green]✓[/green] wrote {out_base} ({len(df):,} rows)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="headline-body-sentiment", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run-all", help="Run canonical build → sentiment → gap → stats → figures (optional bias)")
    run.add_argument("--mode", choices=["summary", "fulltext"], default="summary")
    run.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    run.add_argument("--bias-csv", default="", help="Optional bias table CSV path to merge (URL-based or domain-based).")
    run.add_argument("--vader", action="store_true", help="Also compute VADER sentiment as a fast robustness baseline.")
    run.add_argument("--min-group-n", type=int, default=30)
    run.set_defaults(func=cmd_run_all)

    b = sub.add_parser("build", help="Build the canonical dataset only.")
    b.add_argument("--mode", choices=["summary", "fulltext"], default="summary")
    b.set_defaults(func=cmd_build)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # normalize optional args
    if getattr(args, "bias_csv", None) is not None and args.bias_csv == "":
        args.bias_csv = None

    args.func(args)


if __name__ == "__main__":
    main()



from __future__ import annotations

from pathlib import Path

from nlp_headline_body_sentiment.datasets import build_canonical_dataset
from nlp_headline_body_sentiment.paths import ProjectPaths, find_repo_root


def main() -> None:
    root = find_repo_root()
    paths = ProjectPaths(root)
    gvfc = paths.data_dir / "GVFC_extension_multimodal.csv"
    assert gvfc.exists(), f"Missing {gvfc}"

    df = build_canonical_dataset(gvfc, body_mode="summary", full_text_dir=paths.full_text_dir)
    assert len(df) > 1000, "Unexpectedly small dataset; input may be wrong."
    assert {"id", "url", "headline_text", "body_text", "headline_clean", "body_clean"}.issubset(df.columns)

    # Basic sanity: no empty URLs and IDs are unique-ish
    assert (df["url"].astype(str).str.strip() != "").all()
    assert df["id"].nunique() == len(df)

    print("smoke_check OK:", len(df), "rows")


if __name__ == "__main__":
    main()



from __future__ import annotations

import csv
from pathlib import Path

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    gvfc = data_dir / "GVFC_extension_multimodal.csv"
    assert gvfc.exists(), f"Missing {gvfc}"

    with gvfc.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        required = {
            "id",
            "article_url",
            "headline",
            "presum_summary_of_full_article_text",
            "Q2 Focus",
            "Q3 Theme1",
            "Q3 Theme2",
        }
        assert r.fieldnames is not None, "CSV has no header row"
        missing = sorted(required - set(r.fieldnames))
        assert not missing, f"GVFC master missing columns: {missing}"

        ids: set[str] = set()
        n = 0
        empty_url = 0
        for row in r:
            n += 1
            ids.add(row.get("id", ""))
            if not (row.get("article_url", "") or "").strip():
                empty_url += 1

        assert n > 1000, f"Unexpectedly small GVFC master: {n} rows"
        assert empty_url == 0, f"Found {empty_url} empty article_url values"
        assert len(ids) == n, f"Expected unique ids; got {len(ids)} unique over {n} rows"

    print("smoke_check OK:", n, "rows")


if __name__ == "__main__":
    main()



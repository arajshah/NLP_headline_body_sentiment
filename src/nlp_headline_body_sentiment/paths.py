from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized paths to keep notebooks/scripts consistent."""

    repo_root: Path

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def notebooks_dir(self) -> Path:
        return self.repo_root / "notebooks"

    @property
    def derived_dir(self) -> Path:
        return self.data_dir / "derived"

    @property
    def external_dir(self) -> Path:
        return self.data_dir / "external"

    @property
    def reports_dir(self) -> Path:
        return self.repo_root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def full_text_dir(self) -> Path:
        return self.data_dir / "full_text"

    @property
    def raw_html_dir(self) -> Path:
        return self.data_dir / "raw_html"

    def ensure_dirs(self) -> None:
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)


def find_repo_root(start: Path | None = None) -> Path:
    """Find repo root by walking upward until we see data/ and notebooks/."""
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "data").exists() and (p / "notebooks").exists():
            return p
    # fallback: current directory
    return cur



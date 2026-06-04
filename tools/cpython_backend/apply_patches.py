from __future__ import annotations

from pathlib import Path
import subprocess

from .fetch_cpython import cpython_source_dir


def patch_files(patch_dir: Path) -> list[Path]:
    return sorted(patch_dir.glob("*.patch"))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = cpython_source_dir(repo_root)
    for patch_file in patch_files(repo_root / "runtimes" / "cpython" / "patches"):
        subprocess.run(["git", "-C", str(source_dir), "apply", str(patch_file)], check=True)


if __name__ == "__main__":
    main()

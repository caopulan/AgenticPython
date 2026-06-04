from __future__ import annotations

from pathlib import Path
import subprocess


def cpython_source_dir(repo_root: Path) -> Path:
    return repo_root / ".agentpython-build" / "cpython"


def build_fetch_commands(repo_root: Path, *, tag: str = "v3.12.13") -> list[list[str]]:
    destination = cpython_source_dir(repo_root)
    if destination.exists():
        return [
            [
                "git",
                "-C",
                str(destination),
                "fetch",
                "--tags",
                "--depth",
                "1",
                "origin",
                tag,
            ]
        ]
    return [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            tag,
            "https://github.com/python/cpython.git",
            str(destination),
        ]
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cpython_source_dir(repo_root).parent.mkdir(parents=True, exist_ok=True)
    for command in build_fetch_commands(repo_root):
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

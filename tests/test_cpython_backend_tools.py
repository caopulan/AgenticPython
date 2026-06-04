from tools.cpython_backend.apply_patches import patch_files
from tools.cpython_backend.fetch_cpython import build_fetch_commands, cpython_source_dir


def test_cpython_source_dir_lives_in_ignored_build_cache(tmp_path):
    assert cpython_source_dir(tmp_path) == tmp_path / ".agentpython-build" / "cpython"


def test_fetch_commands_pin_cpython_312_tag(tmp_path):
    commands = build_fetch_commands(tmp_path, tag="v3.12.13")

    assert commands[0] == [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "v3.12.13",
        "https://github.com/python/cpython.git",
        str(tmp_path / ".agentpython-build" / "cpython"),
    ]


def test_patch_files_are_sorted(tmp_path):
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "0002-second.patch").write_text("second", encoding="utf-8")
    (patch_dir / "0001-first.patch").write_text("first", encoding="utf-8")

    assert patch_files(patch_dir) == [
        patch_dir / "0001-first.patch",
        patch_dir / "0002-second.patch",
    ]

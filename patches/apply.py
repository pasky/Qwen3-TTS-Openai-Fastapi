#!/usr/bin/env python3
"""Apply tracked patches to installed venv packages.

Usage:
    uv run python patches/apply.py          # apply all patches
    uv run python patches/apply.py --check  # check if patches are already applied

Run this after `uv sync` or `pip install` whenever vllm-omni is (re)installed.
See docs/vllm-backend.md § "Required Patches" for full details.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

PATCHES_DIR = Path(__file__).parent

# Map: patch filename -> (package import name, relative path inside site-packages)
PATCH_TARGETS: dict[str, tuple[str, str]] = {
    "vllm_omni_qwen3_tts.patch": (
        "vllm_omni",
        "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts.py",
    ),
}


def _find_site_packages_dir(package: str) -> Path | None:
    """Find the site-packages directory containing *package*."""
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return None
    # e.g. .venv/lib/python3.12/site-packages/vllm_omni/__init__.py
    pkg_init = Path(spec.origin)
    # Walk up to site-packages
    for parent in pkg_init.parents:
        if parent.name == "site-packages":
            return parent
    return None


def apply_patch(patch_file: Path, site_packages: Path, *, check_only: bool = False) -> bool:
    """Apply a unified diff patch. Returns True if patch was applied (or already applied)."""
    target_rel = PATCH_TARGETS[patch_file.name][1]
    target_file = site_packages / target_rel

    if not target_file.exists():
        print(f"  SKIP  {patch_file.name}: target {target_file} not found")
        return False

    # Check if already applied (patch --reverse --dry-run succeeds)
    result = subprocess.run(
        ["patch", "--reverse", "--dry-run", "-p0", "-i", str(patch_file)],
        cwd=site_packages,
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"  OK    {patch_file.name}: already applied")
        return True

    if check_only:
        print(f"  NEED  {patch_file.name}: not yet applied")
        return False

    # Check if patch applies cleanly (forward dry-run)
    result = subprocess.run(
        ["patch", "--dry-run", "-p0", "-i", str(patch_file)],
        cwd=site_packages,
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  FAIL  {patch_file.name}: patch does not apply cleanly")
        print(f"        stdout: {result.stdout.decode()}")
        print(f"        stderr: {result.stderr.decode()}")
        return False

    # Apply for real
    result = subprocess.run(
        ["patch", "-p0", "-i", str(patch_file)],
        cwd=site_packages,
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"  DONE  {patch_file.name}: applied successfully")
        return True
    else:
        print(f"  FAIL  {patch_file.name}: patch command failed")
        print(f"        {result.stderr.decode()}")
        return False


def main() -> int:
    check_only = "--check" in sys.argv

    ok = True
    for patch_name, (package, _rel) in PATCH_TARGETS.items():
        patch_file = PATCHES_DIR / patch_name
        if not patch_file.exists():
            print(f"  MISS  {patch_name}: patch file not found")
            ok = False
            continue

        site_packages = _find_site_packages_dir(package)
        if site_packages is None:
            print(f"  SKIP  {patch_name}: package '{package}' not installed")
            continue

        if not apply_patch(patch_file, site_packages, check_only=check_only):
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

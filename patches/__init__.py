"""Tracked patches for third-party packages.

Patch files (.patch) are applied to installed site-packages after `uv sync`:

    uv run python patches/apply.py          # apply all patches
    uv run python patches/apply.py --check  # verify patch status

See each .patch file for details on what it fixes and why.
"""

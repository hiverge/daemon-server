"""Remove filesystem entries sandbox execution creates outside the repository."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import common_tools

logger = logging.getLogger(__name__)


def _raise_walk_error(error: OSError) -> None:
  # A partial inventory could mistake an existing file for a newly created one.
  raise error


def _walk(
  roots: list[Path],
  excluded: set[Path],
) -> Generator[tuple[str, list[str], list[str]], None, None]:
  for root in roots:
    for directory, dirs, files in os.walk(
      root,
      topdown=True,
      onerror=_raise_walk_error,
      followlinks=False,
    ):
      parent = Path(directory)
      dirs[:] = [name for name in dirs if parent / name not in excluded]
      files[:] = [name for name in files if parent / name not in excluded]
      yield directory, dirs, files


@contextmanager
def cleanup_new_external_paths(
  repo_dir: Path,
  *,
  root: Path,
) -> Generator[None, None, None]:
  """Inventory paths before execution and remove new external paths afterward.

  Existing entries, including symlinks, are preserved by name. This does not
  roll back edits or deletions of existing files. Kernel-managed filesystems
  are excluded; shared-memory files in /dev/shm are included.

  The caller must stop the command's processes before leaving this context.
  """
  root = root.resolve()
  repo_dir = repo_dir.resolve()
  excluded = {repo_dir, root / "proc", root / "sys", root / "dev"}
  roots = [root] if root != repo_dir else []
  shared_memory = root / "dev" / "shm"
  if shared_memory.is_dir() and not shared_memory.is_relative_to(repo_dir):
    roots.append(shared_memory)

  # Only names are saved: large datasets and environments are never copied.
  before = {
    directory: set(dirs + files) for directory, dirs, files in _walk(roots, excluded)
  }

  try:
    yield
  finally:
    try:
      for directory, dirs, files in _walk(roots, excluded):
        existing = before.get(directory, set())
        for name in dirs + files:
          if name in existing:
            continue
          path = Path(directory) / name
          try:
            if path.is_symlink() or not path.is_dir():
              path.unlink(missing_ok=True)
            else:
              shutil.rmtree(path)
          except FileNotFoundError:
            # A process may already have removed its temporary file.
            pass
          if name in dirs:
            # Do not walk a new directory after removing its entire subtree.
            dirs.remove(name)
    except OSError:
      logger.exception("Could not clean up files created outside the repository")
      common_tools.mark_contaminated()
      raise

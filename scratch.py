"""Bounded session scratch transport; the coordinator owns the saved state."""

import base64
import binascii
import os
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

# Keep the wire format and limits in sync with hive/virtual_workspace/scratch.py.
SCRATCH_DIR = Path("/scratch")
MAX_SCRATCH_BYTES = 10 * 1024 * 1024
MAX_SCRATCH_FILES = 1024
MAX_SCRATCH_ENTRIES = 2048
MAX_SCRATCH_PATH_BYTES = 1024


class ScratchError(ValueError):
  """An invalid or oversized scratch tree, safe to describe to the agent."""


def _validate_path(path: str) -> None:
  if (
    not isinstance(path, str)
    or not path
    or PurePosixPath(path).is_absolute()
    or any(part in ("", ".", "..") for part in path.split("/"))
    or "\0" in path
    or len(path.encode("utf-8")) > MAX_SCRATCH_PATH_BYTES
  ):
    raise ScratchError("Invalid scratch file path.")


def _check_size(count: int, size: int) -> None:
  if count > MAX_SCRATCH_FILES:
    raise ScratchError(f"Scratch storage is limited to {MAX_SCRATCH_FILES} files.")
  if size > MAX_SCRATCH_BYTES:
    raise ScratchError("Scratch storage exceeds the 10 MiB limit.")


def clear() -> None:
  """Remove prior shell state before evaluation or another session's command."""
  if SCRATCH_DIR.is_symlink():
    SCRATCH_DIR.unlink()
  elif SCRATCH_DIR.is_dir():
    shutil.rmtree(SCRATCH_DIR)
  elif SCRATCH_DIR.exists():
    SCRATCH_DIR.unlink()


def _decode(files: dict) -> dict[str, tuple[bytes | None, int]]:
  if not isinstance(files, dict):
    raise ScratchError("scratch_files must be a mapping.")
  decoded = {}
  size = 0
  file_count = 0
  all_paths = set()
  for name, value in files.items():
    _validate_path(name)
    if (
      not isinstance(value, dict)
      or "content" not in value
      or (value["content"] is not None and not isinstance(value["content"], str))
      or type(value.get("mode")) is not int
    ):
      raise ScratchError("Invalid scratch file data.")
    all_paths.add(name)
    all_paths.update(
      parent.as_posix()
      for parent in PurePosixPath(name).parents
      if parent != PurePosixPath(".")
    )
    if len(all_paths) > MAX_SCRATCH_ENTRIES:
      raise ScratchError("Scratch storage contains too many directory entries.")
    if value["content"] is None:
      decoded[name] = (None, 0o700)
      continue
    if len(value["content"]) > 4 * ((MAX_SCRATCH_BYTES + 2) // 3):
      raise ScratchError("Scratch storage exceeds the 10 MiB limit.")
    try:
      content = base64.b64decode(value["content"], validate=True)
    except (ValueError, binascii.Error):
      raise ScratchError("Invalid scratch file encoding.") from None
    size += len(content)
    file_count += 1
    _check_size(file_count, size)
    decoded[name] = (content, 0o600 | (value["mode"] & 0o111))
  for name in decoded:
    if any(
      parent.as_posix() in decoded and decoded[parent.as_posix()][0] is not None
      for parent in PurePosixPath(name).parents
    ):
      raise ScratchError("A scratch path cannot be both a file and directory.")
  return decoded


@contextmanager
def restored(files: dict):
  """Install one session's snapshot and always remove it before releasing a pod."""
  clear()
  try:
    decoded = _decode(files)
    SCRATCH_DIR.mkdir(mode=0o700)
    for name, (content, mode) in decoded.items():
      path = SCRATCH_DIR / name
      if content is None:
        path.mkdir(parents=True, exist_ok=True)
        continue
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_bytes(content)
      path.chmod(mode)
    yield
  finally:
    clear()


def changes(previous: dict) -> dict:
  """Collect regular files and deletions after a command, including failed ones.

  Reject links and special files rather than following them outside scratch or
  blocking on a pipe. Nothing is committed if any file exceeds the limits.
  """
  current = {}
  size = 0
  file_count = 0
  if SCRATCH_DIR.is_symlink():
    raise ScratchError("Scratch storage cannot be a symbolic link.")
  if SCRATCH_DIR.exists():
    if not SCRATCH_DIR.is_dir():
      raise ScratchError("/scratch must remain a directory.")
    directories = [SCRATCH_DIR]
    entry_count = 0
    while directories:
      directory = directories.pop()
      with os.scandir(directory) as entries:
        for entry in entries:
          entry_count += 1
          # Bound traversal as well as bytes, including empty directory trees.
          if entry_count > MAX_SCRATCH_ENTRIES:
            raise ScratchError("Scratch storage contains too many directory entries.")
          name = Path(entry.path).relative_to(SCRATCH_DIR).as_posix()
          _validate_path(name)
          if entry.is_symlink():
            raise ScratchError("Symbolic links are not supported in scratch storage.")
          if entry.is_dir(follow_symlinks=False):
            directories.append(Path(entry.path))
            current[name] = {"content": None, "mode": 0o700}
            continue
          if not entry.is_file(follow_symlinks=False):
            raise ScratchError("Only regular files are supported in scratch storage.")
          flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
          with os.fdopen(os.open(entry.path, flags), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
              raise ScratchError("Only regular files are supported in scratch storage.")
            _check_size(file_count + 1, size + info.st_size)
            content = stream.read(MAX_SCRATCH_BYTES - size + 1)
          size += len(content)
          file_count += 1
          _check_size(file_count, size)
          current[name] = {
            "content": base64.b64encode(content).decode("ascii"),
            "mode": 0o600 | (stat.S_IMODE(info.st_mode) & 0o111),
          }
  result = {
    name: value for name, value in current.items() if previous.get(name) != value
  }
  result.update({name: None for name in previous if name not in current})
  return result

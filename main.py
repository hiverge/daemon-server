"""A simple Python sandbox server that executes Python functions."""

# Needed for Python 3.8 type hint compatibility
from __future__ import annotations

import array
import ctypes
import errno
import fcntl
import logging
import os
import shutil
import stat
import struct
import subprocess
import sys
import termios
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request
from pythonjsonlogger import json as json_logger

import common_tools

REPO_DIR = os.environ.get(
  "REPO_DIR", "/app/"
)  # Directory where the repository is mounted
BACKUP_DIR = os.environ.get(
  "BACKUP_DIR", "/shared/repo/"
)  # Backup directory to restore original state
TRACKING_GIT_DIR = os.environ.get(
  "TRACKING_GIT_DIR", "/shared/daemon-state/git"
)  # Prepared post-setup Git baseline, outside the temporary cleanup roots.
REQUEST_TRACKING_GIT_DIR = f"{TRACKING_GIT_DIR}.request"
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch")
SCRATCH_TRACKING_GIT_DIR = f"{TRACKING_GIT_DIR}.scratch"
# Where bundled helper scripts are exposed inside the sandbox/shell
SCRIPTS_DIR = os.environ.get("SCRIPTS_DIR", "/opt/hiverge/scripts")

# Helper scripts bundled into the freeze (see daemon-server.spec `datas`) and
# copied into SCRIPTS_DIR on startup so they are on hand in an interactive shell.
BUNDLED_SCRIPTS = ("run_multi_evaluator.py",)

app = Flask(__name__)
sandbox_lock = threading.Lock()

# Configure logging
log_handler = logging.StreamHandler()
formatter = json_logger.JsonFormatter(
  fmt="%(asctime)s %(levelname)s %(message)s",
  rename_fields={"levelname": "level"},
  defaults={"category": "system"},
)
log_handler.setFormatter(formatter)
logging.root.handlers = [log_handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TemporaryDirectoryBaseline:
  """A setup directory's inode, immediate names, and indexed parent."""

  identity: tuple[int, int]
  entries: frozenset[str]
  parent: Path | None = None


# Event bits and record layout from Linux UAPI's <linux/inotify.h>.
_IN_MODIFY = 0x00000002
_IN_ATTRIB = 0x00000004
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_UNMOUNT = 0x00002000
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_ONLYDIR = 0x01000000
_IN_EVENT = struct.Struct("iIII")
_IN_WATCH_LOST = _IN_DELETE_SELF | _IN_MOVE_SELF | _IN_UNMOUNT | _IN_IGNORED
_IN_ADDITIONS = _IN_CREATE | _IN_MOVED_TO
_IN_MUTATIONS = _IN_MODIFY | _IN_ATTRIB | _IN_MOVED_FROM | _IN_DELETE | _IN_ADDITIONS


class _TemporaryDirectoryEvents:
  """Select cleanup directories and observe changes to setup entries.

  Watches are installed once during startup inventory. Lost coverage closes the
  event stream and records contamination; cleanup stops because directory scans
  cannot establish whether setup file contents changed during the gap. Events
  arriving during ordinary cleanup remain queued for the next drain.
  """

  def __init__(self, fd: int, add_watch):
    self.fd: int | None = fd
    self._add_watch = add_watch
    self.paths: dict[int, Path] = {}

  @classmethod
  def open(cls) -> _TemporaryDirectoryEvents:
    """Open Linux event tracking; initialization errors abort daemon startup."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [
      ctypes.c_int,
      ctypes.c_char_p,
      ctypes.c_uint32,
    ]
    libc.inotify_add_watch.restype = ctypes.c_int
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
      error = ctypes.get_errno()
      raise OSError(error, os.strerror(error))
    return cls(fd, libc.inotify_add_watch)

  def watch(self, path: Path, directory_fd: int) -> None:
    """Register a startup watch on the already-opened directory."""
    # The proc fd link refers to the directory already opened without following
    # symlinks. Watching it avoids resolving a mutable path again.
    descriptor = self._add_watch(
      self.fd,
      os.fsencode(f"/proc/self/fd/{directory_fd}"),
      _IN_MUTATIONS | _IN_WATCH_LOST | _IN_ONLYDIR,
    )
    if descriptor < 0:
      error = ctypes.get_errno()
      raise OSError(error, os.strerror(error), str(path))
    self.paths[descriptor] = path

  def changed_paths(self) -> set[Path] | None:
    """Record setup mutations and select parents that may contain additions.

    Event names are compared with the immutable startup inventory. Writes and
    removals of new names are expected while producing request artifacts.
    Mutations of startup names flag contamination without scanning their parent.
    Directory self-events have no name and describe the watched setup inode.
    Lost coverage records its reason, closes the watcher, and returns None.
    """
    if self.fd is None:
      _record_filesystem_contamination("Filesystem event tracking is unavailable.")
      return None
    dirty = set()
    try:
      # FIONREAD supplies the actual queued byte count. One queue snapshot keeps
      # concurrent writers from extending this read loop indefinitely.
      queued = array.array("i", [0])
      fcntl.ioctl(self.fd, termios.FIONREAD, queued, True)
      if not queued[0]:
        return dirty
      data = os.read(self.fd, queued[0])
      offset = 0
      while offset < len(data):
        descriptor, mask, _cookie, length = _IN_EVENT.unpack_from(data, offset)
        offset += _IN_EVENT.size
        name = os.fsdecode(data[offset : offset + length].split(b"\0", 1)[0])
        offset += length
        if offset > len(data):
          raise ValueError("Incomplete inotify event")
        if mask & _IN_Q_OVERFLOW:
          _record_filesystem_contamination("Filesystem event queue overflowed.")
          logger.warning("Filesystem event queue overflowed; stopping cleanup.")
          self.close()
          return None
        elif mask & _IN_WATCH_LOST:
          directory = self.paths.get(descriptor)
          _record_filesystem_contamination(
            f"Setup directory changed or its watch was lost: {directory}."
          )
          logger.warning(
            "Setup directory changed or its watch was lost: %s; stopping cleanup.",
            directory,
          )
          self.close()
          return None
        elif descriptor in self.paths:
          directory = self.paths[descriptor]
          path = directory / name if name else directory
          if path in _temporary_virtual or any(
            path.is_relative_to(protected) for protected in _temporary_excluded
          ):
            continue
          baseline = _temporary_baseline.get(directory)
          if baseline is not None and (not name or name in baseline.entries):
            if mask & _IN_MUTATIONS:
              _record_filesystem_contamination(
                f"Setup filesystem state changed: {path}."
              )
          if mask & _IN_ADDITIONS:
            dirty.add(directory)
      return dirty
    except (OSError, ValueError, struct.error) as exc:
      _record_filesystem_contamination(f"Filesystem event tracking failed: {exc}.")
      logger.warning("Filesystem event tracking failed; stopping cleanup: %s", exc)
      self.close()
      return None

  def close(self) -> None:
    if self.fd is not None:
      os.close(self.fd)
      self.fd = None
    self.paths.clear()


# Ordinary filesystem trees are indexed from /; temporary roots also cover
# /dev/shm beneath the excluded device tree.
_ROOT_DIR = Path("/")
_TEMPORARY_DIRS = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))
# Populated once at startup. Importing this module leaves cleanup disabled.
_temporary_baseline: dict[Path, _TemporaryDirectoryBaseline] = {}
_temporary_events: _TemporaryDirectoryEvents | None = None
_temporary_excluded: frozenset[Path] = frozenset()
_temporary_virtual: frozenset[Path] = frozenset()
# The execution gate and retirement hook consume this sticky reason.
_filesystem_contamination_reason: str | None = None


def _record_filesystem_contamination(reason: str) -> None:
  """Keep and log the first mutation, cleanup failure, or coverage failure.

  The reason lasts until the daemon captures its startup inventory again.
  Detection uses filesystem events and performs no content hashing. Directory
  watches have coverage limits, including mmap writes and file bind mounts.
  """
  global _filesystem_contamination_reason
  if _filesystem_contamination_reason is None:
    _filesystem_contamination_reason = reason
    logger.warning("Filesystem contamination: %s", reason)


@contextmanager
def _open_temporary_directory(
  path: Path,
  baseline: dict[Path, _TemporaryDirectoryBaseline],
  parent: Path | None = None,
):
  """Open through indexed ancestors, checking their identities and symlinks.

  A matching leaf inode can be reached through a replaced parent symlink.
  Opening each component relative to the previous fd checks the entire chain.
  """
  chain = [path]
  ancestor = baseline[path].parent if path in baseline else parent
  while ancestor is not None:
    chain.append(ancestor)
    ancestor = baseline[ancestor].parent
  directory_fd = None
  try:
    for component in reversed(chain):
      next_fd = os.open(
        component if directory_fd is None else component.name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=directory_fd,
      )
      if directory_fd is not None:
        os.close(directory_fd)
      directory_fd = next_fd
      if component in baseline:
        info = os.fstat(directory_fd)
        if (info.st_dev, info.st_ino) != baseline[component].identity:
          raise OSError(errno.ESTALE, "Setup directory was replaced", str(component))
    yield directory_fd
  finally:
    if directory_fd is not None:
      os.close(directory_fd)


def _capture_temporary_baseline() -> dict[Path, _TemporaryDirectoryBaseline]:
  """Inventory cleanup directories once, before serving requests.

  Each watch is installed before reading the directory's names. Repository and
  scratch workspaces, backup trees and Git tracking paths have their own request
  lifecycle and are excluded. Kernel and device trees are skipped; /dev/shm is
  indexed separately with the temporary roots.
  Symlinked directories are retained as entries and are never traversed.
  Only directory names and identities are recorded; file contents are not read.
  New trees are removed through their first new entry in an indexed parent.
  Absent temporary roots are allowed. Other initialization and inventory errors
  propagate, closing the watcher before the daemon can serve requests.
  """
  global _temporary_events, _temporary_excluded, _temporary_virtual
  global _filesystem_contamination_reason
  _filesystem_contamination_reason = None
  if _temporary_events is not None:
    _temporary_events.close()
  _temporary_events = None
  events = _TemporaryDirectoryEvents.open()
  try:
    baseline = {}
    excluded = {
      Path(path).resolve()
      for path in (
        REPO_DIR,
        BACKUP_DIR,
        SCRATCH_DIR,
        TRACKING_GIT_DIR,
        REQUEST_TRACKING_GIT_DIR,
        SCRATCH_TRACKING_GIT_DIR,
      )
    }
    root_directory = _ROOT_DIR.resolve(strict=True)
    virtual = tuple(root_directory / name for name in ("proc", "sys", "dev"))
    _temporary_excluded = frozenset(excluded)
    _temporary_virtual = frozenset(virtual)
    temporary = []
    for directory in _TEMPORARY_DIRS:
      try:
        path = directory.resolve(strict=True)
      except FileNotFoundError:
        # Images may omit temporary roots such as /dev/shm.
        continue
      temporary.append(path)
    pending = [(path, None) for path in temporary]
    pending.append((root_directory, None))
    seen = set()
    while pending:
      path, parent = pending.pop()
      if any(path.is_relative_to(protected) for protected in excluded):
        continue
      if any(path.is_relative_to(tree) for tree in virtual) and not any(
        path.is_relative_to(directory) for directory in temporary
      ):
        continue
      with _open_temporary_directory(path, baseline, parent) as directory_fd:
        info = os.fstat(directory_fd)
        identity = (info.st_dev, info.st_ino)
        # Directory aliases and bind mounts can revisit an already indexed inode.
        if identity in seen:
          continue
        events.watch(path, directory_fd)
        names = set()
        children = []
        with os.scandir(directory_fd) as entries:
          for entry in entries:
            names.add(entry.name)
            if entry.is_dir(follow_symlinks=False):
              children.append(path / entry.name)
        # Reserve lifecycle and virtual-tree names even if they are created
        # after inventory. Their owning subsystem controls their contents.
        for protected in (*excluded, *virtual):
          if protected != path and protected.is_relative_to(path):
            names.add(protected.relative_to(path).parts[0])
        baseline[path] = _TemporaryDirectoryBaseline(identity, frozenset(names), parent)
        seen.add(identity)
        pending.extend((child, path) for child in children)
  except BaseException:
    # Release startup resources while preserving the original failure.
    events.close()
    raise
  _temporary_events = events
  return baseline


def _cleanup_temporary_entries() -> None:
  """Delete names absent from the startup baseline, logging cleanup failures.

  Events select changed setup directories. Unreliable coverage stops cleanup.
  New trees are removed recursively and symlinks are unlinked; existing file
  changes are recorded and their contents are left as written. Failed removals
  record contamination without retrying.
  """
  if _temporary_events is None:
    return
  changed = _temporary_events.changed_paths()
  if changed is None:
    return
  for root in changed:
    if root not in _temporary_baseline:
      continue
    baseline = _temporary_baseline[root]
    try:
      with _open_temporary_directory(root, _temporary_baseline) as directory_fd:
        with os.scandir(directory_fd) as entries:
          for entry in entries:
            if entry.name in baseline.entries:
              continue
            try:
              if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.name, dir_fd=directory_fd)
              else:
                os.unlink(entry.name, dir_fd=directory_fd)
            except FileNotFoundError:
              # A writer may have removed its own temporary entry already.
              continue
            except OSError as exc:
              _record_filesystem_contamination(
                f"New filesystem entry survived cleanup at {root / entry.name}: {exc}."
              )
              logger.warning(
                "Could not remove new filesystem entry %s: %s", root / entry.name, exc
              )
    except OSError as exc:
      _record_filesystem_contamination(
        f"Could not clean setup directory {root}: {exc}."
      )
      logger.warning("Could not clean setup directory %s: %s", root, exc)


def _is_sandbox_contaminated() -> bool:
  """Share the retirement decision between admission and response handling."""
  return _filesystem_contamination_reason is not None or common_tools.is_contaminated()


def lock_sandbox():
  """Serialize execution and refuse reuse after contamination.

  Check contamination after acquiring the lock, so the preceding request has
  finished cleanup before the decision. A request that contaminates the worker
  keeps its original response; subsequent requests are refused before their
  handler or cleanup runs. The lock is released on every exit path.
  """

  def decorator(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
      if not sandbox_lock.acquire(blocking=False):
        logger.info("Experiment already running on sandbox. Rejecting new request.")
        return jsonify(
          {"output": None, "metainfo": "Only one experiment can run at a time."}
        ), 429

      try:
        if _is_sandbox_contaminated():
          logger.info("Sandbox is contaminated. Rejecting new execution request.")
          return jsonify(
            {
              "status": "failed",
              "error": "Sandbox is contaminated and cannot accept further execution.",
            }
          ), 503
        try:
          with common_tools.request_process_cleanup():
            return f(*args, **kwargs)
        finally:
          # Cleanup completes before the lock becomes available to the next
          # request, including when the handler raises.
          _cleanup_temporary_entries()
      finally:
        sandbox_lock.release()

    return decorated_function

  return decorator


# The header lets the coordinator retain the result and stop using this worker.
SANDBOX_RETIRED_HEADER = "X-Hive-Sandbox-Retired"
# Retain the existing response-flush grace. This allows time for WSGI to send
# the response; it cannot guarantee the caller has received it.
_RETIREMENT_GRACE = 1.0
_retirement_timer: threading.Timer | None = None


def _retire_sandbox() -> None:
  """Log why the sandbox is unsafe and exit with the existing failure status.

  The daemon runs as PID 1, so exiting also ends surviving container processes.
  The operator must replace the whole pod: restarting its container preserves
  per-pod volumes, including /shared.
  """
  logger.warning(
    "Exiting contaminated sandbox for replacement.",
    extra={
      "filesystem_contamination_reason": _filesystem_contamination_reason,
      "process_cleanup_failed": common_tools.is_contaminated(),
    },
  )
  # `os._exit` skips interpreter cleanup, which would join the abandoned readers.
  os._exit(1)


@app.after_request
def _retire_if_contaminated(response):
  """Flag retirement while preserving the completed response's body and status.

  Cleanup runs before this hook. The coordinator can keep the result and remove
  the worker from its pool as soon as it sees the header. Subsequent execution
  is refused by the admission gate while the response-flush grace runs.
  """
  global _retirement_timer
  if _is_sandbox_contaminated():
    response.headers[SANDBOX_RETIRED_HEADER] = "true"
    # A health check or busy response can observe contamination during another
    # request's cleanup. Let that request's response schedule exit after cleanup.
    if sandbox_lock.acquire(blocking=False):
      try:
        # Repeated responses share one timer, guarded by the request lock.
        if _retirement_timer is None:
          try:
            timer = threading.Timer(_RETIREMENT_GRACE, _retire_sandbox)
            timer.start()
          except RuntimeError as exc:
            # Thread exhaustion can prevent startup. Preserve this response and
            # leave scheduling available for the next response to retry.
            logger.warning("Could not schedule sandbox retirement: %s", exc)
          else:
            _retirement_timer = timer
      finally:
        sandbox_lock.release()
  return response


def _resolve_repo_path(
  rel_path: str, dir: str, *, follow_final_symlink: bool = True
) -> Path:
  """Resolve an overlay path and require it to remain inside its root.

  Writes follow the final symlink, matching normal file-open behaviour. Deletions
  resolve only the parent so unlinking removes the named symlink itself.
  """
  root = Path(dir).resolve()
  requested_path = root / rel_path
  full_path = (
    requested_path.resolve()
    if follow_final_symlink
    else requested_path.parent.resolve() / requested_path.name
  )
  if not full_path.is_relative_to(root):
    raise ValueError(f"Path escapes repo directory: {rel_path}")
  return full_path


def _apply_code_files(code_files: dict[str, str | None], dir: str) -> None:
  """Lay the request's files over the supplied directory, in order.

  Content writes the file, creating missing parent directories; None deletes
  it. Shared by /run_code and /shell so both accept the same files

  Raises ValueError for a path that escapes the repo or cannot be written
  (e.g. a file where a directory is), which both endpoints return as a JSON 400.
  """
  for rel_path, content in code_files.items():
    try:
      if content is None:
        full_path = _resolve_repo_path(rel_path, dir, follow_final_symlink=False)
        full_path.unlink(missing_ok=True)
        continue
      full_path = _resolve_repo_path(rel_path, dir)
      full_path.parent.mkdir(parents=True, exist_ok=True)
      full_path.write_text(content, encoding="utf-8")
    except OSError as e:
      raise ValueError(f"Cannot write code file {rel_path}: {e}") from e


@contextmanager
def _scratch_workspace(files: dict[str, str]):
  """Materialize a request's scratch snapshot and clear it before worker reuse."""
  root = Path(SCRATCH_DIR)

  def reset():
    _remove_tracking_directory(SCRATCH_TRACKING_GIT_DIR)
    if root.is_dir() and not root.is_symlink():
      shutil.rmtree(root)
    else:
      root.unlink(missing_ok=True)
    root.mkdir(parents=True)

  reset()
  try:
    _apply_code_files(files, SCRATCH_DIR)
    yield
  finally:
    reset()


def execute_python_function(
  code_files: dict[str, str | None],
  args: list,
  timeout: float,
  evaluation_script: str,
) -> str:
  """Execute a Python function in a temporary directory."""
  # Restore the original repository state using rsync
  subprocess.run(["rsync", "-a", "--delete", BACKUP_DIR, REPO_DIR])

  # Stringify only. The command below goes to `Popen` as a list with no shell, so
  # nothing splits on whitespace and nothing strips quotes: adding them would put
  # the quote characters inside `sys.argv[n]`.
  args = [str(arg) for arg in args]

  _apply_code_files(code_files, REPO_DIR)

  # Run the Python program
  # Use system python3 instead of sys.executable to avoid issues with PyInstaller bundles
  # TODO: if people write evaluation_script with a different language extension,
  # we should detect that and choose the right executable (e.g. node for .js)
  python_executable = os.environ.get("PYTHON_EXECUTABLE", "python3")
  try:
    output = common_tools.run_command(
      [python_executable, evaluation_script] + args,
      REPO_DIR,
      timeout,
      log_output=True,
    )
    return output
  except common_tools.FunctionExecutionError as e:
    logger.info("Run command failed: %s. Attempting to read checkpoint data.", e)
    try:
      # If the script leaves checkpointed json data, find and return it
      output = common_tools.run_command(["cat", "checkpoint.json"], REPO_DIR)
      return jsonify(
        {
          "status": "success",
          "data": {"output": output, "metainfo": "Checkpoint"},
        }
      ), 200
    except common_tools.FunctionExecutionError as ee:
      logger.info("Failed to read checkpoint data: %s. Returning original error.", ee)
      # Re-raise the original error unchanged. `run_command` already produces
      # the final, user-facing message (including any "Execution failed:"
      # prefix), so it must not be wrapped again here.
      raise e


@app.route("/health", methods=["GET"])
def health_check():
  """Health check endpoint."""
  return jsonify({"status": "healthy"}), 200


@app.route("/run_code", methods=["POST"])
@lock_sandbox()
def run_function():
  """Run the Python function provided in the request."""
  try:
    if not request.is_json:
      logger.error("Request content type is not application/json")
      return jsonify(
        {"status": "failed", "error": "Content-Type must be application/json"}
      ), 400

    code = request.json.get("code")
    timeout = float(request.json.get("timeout"))
    args = request.json.get("args", ())
    evaluation_script = request.json.get("evaluation_script", "evaluator.py")

    logger.info(
      "Executing code with timeout=%s,  evaluation_script=%s",
      timeout,
      evaluation_script,
    )

    result = execute_python_function(code, args, timeout, evaluation_script)
    return result, 200

  except ValueError as e:
    logger.error("Invalid request: %s", e)
    return jsonify({"status": "failed", "error": str(e)}), 400
  except common_tools.FunctionExecutionError as e:
    logger.error("Function execution failed: %s", e)
    return jsonify({"status": "failed", "error": str(e)}), 400
  except subprocess.SubprocessError as e:
    logger.error("Unexpected error: %s", e)
    if str(e) == "Exception occurred in preexec_fn.":
      return jsonify(
        {"status": "failed", "error": "Execution failed: Memory limit exceeded"}
      ), 500
    return jsonify({"status": "failed", "error": "Internal server error"}), 500


def _git(args: list[str], dir: str, git_dir: str) -> list[str]:
  """Address the daemon's tracking repo separately from the project's .git."""
  return ["git", "--git-dir", git_dir, "--work-tree", dir] + args


def _run_tracking_git(args: list[str], dir: str, git_dir: str):
  """Run tracking Git with project ignore rules and a fixed configuration.

  A shell can change global or system Git configuration. Tracking uses its own
  repository configuration so those changes cannot carry into later requests.
  """
  env = os.environ.copy()
  env["GIT_CONFIG_GLOBAL"] = "/dev/null"
  env["GIT_CONFIG_SYSTEM"] = "/dev/null"
  result = subprocess.run(
    _git(args, dir, git_dir), capture_output=True, text=True, env=env
  )
  if result.returncode != 0:
    raise RuntimeError(
      f"Tracking git {args[0]} failed: {result.stderr.strip() or result.stdout.strip()}"
    )
  return result


def _remove_tracking_directory(git_dir: str) -> None:
  """Remove a tracking directory, including a replaced file or symlink."""
  path = Path(git_dir)
  if path.is_dir() and not path.is_symlink():
    shutil.rmtree(path)
  else:
    path.unlink(missing_ok=True)


def init_git_tracking(dir: str, git_dir: str):
  """Create clean tracking metadata with no inherited hooks or exclude files."""
  _remove_tracking_directory(git_dir)
  _run_tracking_git(["init", "--template="], dir, git_dir)


def prepare_git_tracking() -> None:
  """Hash the post-setup work tree once, before accepting requests.

  The index is prepared against REPO_DIR so it includes that tree's file
  metadata. Each request copies this template and reads the existing blobs
  through Git's object alternates. Request objects and index updates stay in
  their own directory; tracking never updates this baseline after startup.
  """
  init_git_tracking(REPO_DIR, TRACKING_GIT_DIR)
  # git add alone leaves no index when the setup work tree is empty.
  _run_tracking_git(["read-tree", "--empty"], REPO_DIR, TRACKING_GIT_DIR)
  snapshot_work_tree(REPO_DIR, TRACKING_GIT_DIR)


@contextmanager
def _request_git_tracking():
  """Seed a fresh request repo from the startup index and share baseline blobs.

  Copying the index reuses content hashes and the original stat information.
  Git still validates the restored tree: overlays and files with changed
  metadata can require content reads. New blobs are written into this request's
  object store, which is removed on success, timeout, or failure.
  """
  baseline = Path(TRACKING_GIT_DIR)
  tracking = Path(REQUEST_TRACKING_GIT_DIR)
  _remove_tracking_directory(REQUEST_TRACKING_GIT_DIR)
  try:
    (tracking / "objects" / "info").mkdir(parents=True)
    (tracking / "refs").mkdir()
    for name in ("HEAD", "config", "index"):
      shutil.copyfile(baseline / name, tracking / name)
    # Git reads existing objects here and writes new objects under tracking.
    (tracking / "objects" / "info" / "alternates").write_text(
      f"{(baseline / 'objects').resolve()}\n", encoding="utf-8"
    )
    yield
  finally:
    _remove_tracking_directory(REQUEST_TRACKING_GIT_DIR)


def snapshot_work_tree(dir: str, git_dir: str):
  """Stage snapshot A, incorporating the request's files into its own index.

  Git's full-tree metadata check preserves its handling of ignore rules,
  symlinks, and deletions. This remains a repository-wide operation; copying
  the startup index saves rebuilding its hashes from an empty index.
  """
  _run_tracking_git(["add", "-A"], dir, git_dir)


def get_changed_files(dir: str, git_dir: str):
  """Return text changes relative to the pre-command index; None is a deletion."""
  # This git command returns all untracked files and files with unstaged changes
  args = ["ls-files", "-z", "--modified", "--others", "--exclude-standard"]
  result = _run_tracking_git(args, dir, git_dir)
  filenames = [name for name in result.stdout.split("\0") if name]

  files = {}
  for fn in filenames:
    try:
      files[fn] = (Path(dir) / fn).read_text(encoding="utf-8")
    except FileNotFoundError:
      files[fn] = None  # deleted by the command
    except (UnicodeDecodeError, ValueError):
      pass  # binary content isn't representable in the JSON response
  return files


@_request_git_tracking()
def execute_shell_command(
  cmd: str,
  cwd: str,
  code_files: dict[str, str | None],
  timeout: float,
) -> tuple[str, dict[str, str | None]]:
  """Execute a shell command.

  Returns the shell command output to give back to the LLM, and a mapping of
  changed relative file paths to their current contents (None for deletions).
  """
  # Restore the original repository state using rsync
  subprocess.run(["rsync", "-a", "--delete", BACKUP_DIR, REPO_DIR])

  _apply_code_files(code_files, REPO_DIR)

  snapshot_work_tree(REPO_DIR, REQUEST_TRACKING_GIT_DIR)
  init_git_tracking(SCRATCH_DIR, SCRATCH_TRACKING_GIT_DIR)
  snapshot_work_tree(SCRATCH_DIR, SCRATCH_TRACKING_GIT_DIR)

  # Run the shell command in its own process group, and time out on its own exit
  # rather than on its output reaching end-of-stream. A backgrounded job (`cmd &`)
  # inherits the pipes, so timing out on end-of-stream would charge the command
  # the full lifetime of everything it spawned.
  stdout_lines: list[str] = []
  stderr_lines: list[str] = []
  process = subprocess.Popen(
    cmd,
    shell=True,
    cwd=cwd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    encoding="utf-8",
    errors="replace",
    start_new_session=True,
  )
  with common_tools.sandboxed_process(process):
    # Shell output is returned to the agent but kept out of the logs: a command
    # like `cat evaluator.py` would otherwise leak source code into them.
    readers = common_tools.start_stream_readers(
      process, stdout_lines, stderr_lines, log_output=False
    )
    try:
      returncode = process.wait(timeout=timeout)
      timed_out = False
    except subprocess.TimeoutExpired:
      returncode = None
      timed_out = True

    # Stop the command and everything it started, then collect whatever output
    # made it through. Undrained output is reported as truncated below: a shell
    # command has no single result line that a leaked process could invalidate.
    common_tools.kill_process_group(process)
    drained = common_tools.join_stream_readers(readers)

  if timed_out:
    output = f"Error: command timed out after {timeout}s"
  else:
    parts = []
    if stdout_lines:
      parts.append("".join(stdout_lines))
    if stderr_lines:
      parts.append(f"STDERR:\n{''.join(stderr_lines)}")
    if returncode != 0:
      parts.append(f"Exit code: {returncode}")
    if not drained:
      parts.append(
        "[Output truncated: the command left background processes holding its output.]"
      )
    output = "\n".join(parts) if parts else "(no output)"

  return output, get_changed_files(REPO_DIR, REQUEST_TRACKING_GIT_DIR)


@app.route("/shell", methods=["POST"])
@lock_sandbox()
def run_shell():
  """Run a shell command from a given file state."""
  try:
    if not request.is_json:
      logger.error("Request content type is not application/json")
      return jsonify(
        {"status": "failed", "error": "Content-Type must be application/json"}
      ), 400

    # Get data and validate required fields
    data = request.get_json()
    missing_fields = [
      field for field in ("cmd", "cwd", "code_files") if field not in data
    ]
    if missing_fields:
      message = f"Missing required field(s): {', '.join(missing_fields)}"
      return jsonify({"status": "failed", "error": message}), 400

    cmd = data["cmd"]
    code_files = data["code_files"]
    timeout = float(data.get("timeout", 120))

    logger.info("Executing shell command with timeout=%s", timeout)

    with _scratch_workspace(data.get("scratch_files", {})):
      cwd = (Path(REPO_DIR) / data["cwd"]).resolve()
      if not any(
        cwd.is_relative_to(Path(root).resolve()) for root in (REPO_DIR, SCRATCH_DIR)
      ):
        raise ValueError("cwd escapes workspace directories")
      output, files = execute_shell_command(cmd, str(cwd), code_files, timeout)
      scratch_files = get_changed_files(SCRATCH_DIR, SCRATCH_TRACKING_GIT_DIR)

    return jsonify(
      {
        "status": "success",
        "result": {
          "output": output,
          "files": files,
          "scratch_files": scratch_files,
        },
      }
    ), 200

  except ValueError as e:
    logger.error("Invalid request: %s", e)
    return jsonify({"status": "failed", "error": str(e)}), 400
  except Exception as e:
    logger.error("Agent execution failed: %s", e)
    return jsonify({"status": "failed", "error": str(e)}), 500


def _bundle_dir() -> Path:
  """Directory holding the bundled data files."""
  return Path(getattr(sys, "_MEIPASS", Path(__file__).parent))


def materialize_shell_scripts() -> None:
  """Copy bundled helper scripts into SCRIPTS_DIR, made executable."""
  try:
    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    for name in BUNDLED_SCRIPTS:
      dst = Path(SCRIPTS_DIR) / name
      shutil.copyfile(_bundle_dir() / name, dst)
      dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    logger.info("Materialized helper scripts into %s", SCRIPTS_DIR)
  except OSError as e:
    logger.warning("Could not materialize helper scripts into %s: %s", SCRIPTS_DIR, e)


if __name__ == "__main__":
  # Ensure required directories exist
  os.makedirs(REPO_DIR, exist_ok=True)
  os.makedirs(BACKUP_DIR, exist_ok=True)
  materialize_shell_scripts()
  prepare_git_tracking()
  _temporary_baseline = _capture_temporary_baseline()
  common_tools.enable_process_cleanup()

  port = int(os.environ.get("PORT", "8080"))

  # Use waitress for production-ready WSGI server
  try:
    from waitress import serve

    logger.info(f"Starting production server on port {port}")
    serve(app, host="0.0.0.0", port=port)
  except ImportError:
    # Fallback to Flask development server if waitress not available
    logger.warning("waitress not found, using Flask development server")
    app.run(debug=False, host="0.0.0.0", port=port)

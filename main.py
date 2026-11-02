"""A simple Python sandbox server that executes Python functions."""

# Needed for Python 3.8 type hint compatibility
from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import sys
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
  """The root inode and immediate entry names recorded after setup."""

  identity: tuple[int, int]
  entries: frozenset[str]


_TEMPORARY_DIRS = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))
# Populated once at startup. Importing this module leaves cleanup disabled.
_temporary_baseline: dict[Path, _TemporaryDirectoryBaseline] = {}


def _capture_temporary_baseline() -> dict[Path, _TemporaryDirectoryBaseline]:
  """Record immediate names in temporary roots before serving requests.

  Setup directories are preserved as whole trees. Reserve configured Git paths
  beneath these roots: the baseline survives requests, and request tracking
  directories are removed by their owning contexts.
  """
  baseline = {}
  for directory in _TEMPORARY_DIRS:
    root = directory.resolve()
    try:
      info = root.stat()
      with os.scandir(root) as entries:
        names = {entry.name for entry in entries}
      for tracking in (
        TRACKING_GIT_DIR,
        REQUEST_TRACKING_GIT_DIR,
        SCRATCH_TRACKING_GIT_DIR,
      ):
        tracking_path = Path(tracking).resolve()
        if tracking_path.is_relative_to(root):
          relative = tracking_path.relative_to(root)
          if relative.parts:
            names.add(relative.parts[0])
      baseline[root] = _TemporaryDirectoryBaseline(
        (info.st_dev, info.st_ino), frozenset(names)
      )
    except FileNotFoundError:
      # Some images have no /dev/shm; capture only roots present at startup.
      continue
    except OSError as exc:
      logger.warning("Could not inventory temporary directory %s: %s", root, exc)
  return baseline


def _cleanup_temporary_entries() -> None:
  """Delete names absent from the startup baseline, logging cleanup failures.

  Only new directory trees are traversed. Changes inside setup directories and
  overwrites of setup files remain in place. Directory-relative operations keep
  deletion anchored to the opened root; symlinks are unlinked.
  """
  for root, baseline in _temporary_baseline.items():
    try:
      directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
      try:
        info = os.fstat(directory_fd)
        if (info.st_dev, info.st_ino) != baseline.identity:
          logger.warning("Skipping cleanup of replaced temporary root %s.", root)
          continue
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
              logger.warning(
                "Could not remove temporary entry %s: %s", root / entry.name, exc
              )
      finally:
        os.close(directory_fd)
    except OSError as exc:
      logger.warning("Could not clean temporary directory %s: %s", root, exc)


def lock_sandbox():
  def decorator(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
      if not sandbox_lock.acquire(blocking=False):
        logger.info("Experiment already running on sandbox. Rejecting new request.")
        return jsonify(
          {"output": None, "metainfo": "Only one experiment can run at a time."}
        ), 429

      try:
        return f(*args, **kwargs)
      finally:
        # Handlers have collected their response and finished process cleanup.
        # Keep the lock until temporary entries are removed so the next request
        # cannot start while this request's filesystem cleanup is running.
        try:
          _cleanup_temporary_entries()
        finally:
          sandbox_lock.release()

    return decorated_function

  return decorator


# How long to let the WSGI server flush the response before exiting.
_RESTART_GRACE = 1.0


def _restart_container() -> None:
  """Exit so the container restarts, discarding anything left running in it.

  The daemon is PID 1 in the sandbox container, so ending it ends the container's
  PID namespace, taking any survivor with it, and the kubelet starts a
  replacement.
  """
  logger.warning(
    "A process survived this evaluation and is still running on the pod. "
    "Restarting the container so it cannot affect later measurements."
  )
  # `os._exit` skips interpreter cleanup, which would join the abandoned readers.
  os._exit(1)


@app.after_request
def _restart_if_contaminated(response):
  """Restart the container once a survivor makes its measurements unreliable."""
  if common_tools.is_contaminated():
    # After the response: this evaluation ran on a clean pod, only later ones are
    # at risk.
    threading.Timer(_RESTART_GRACE, _restart_container).start()
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

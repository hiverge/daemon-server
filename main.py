"""A simple Python sandbox server that executes Python functions."""

# Needed for Python 3.8 type hint compatibility
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from functools import wraps
from pathlib import Path
from typing import Optional

import common_tools
from flask import Flask, jsonify, request
from pythonjsonlogger import json as json_logger

REPO_DIR = os.environ.get("REPO_DIR", "/app/")  # Directory where the repository is mounted
BACKUP_DIR = os.environ.get("BACKUP_DIR", "/shared/repo/")  # Backup directory to restore original state
TRACKING_GIT_DIR = os.environ.get("TRACKING_GIT_DIR", "/tmp/.agent_git")  # Git directory for tracking changes without affecting the actual repo

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


def lock_sandbox():
  def decorator(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
      if not sandbox_lock.acquire(blocking=False):
        logger.info(
          "Experiment already running on sandbox. Rejecting new request."
        )
        return jsonify(
          {"output": None, "metainfo": "Only one experiment can run at a time."}
        ), 429

      try:
        return f(*args, **kwargs)
      finally:
        sandbox_lock.release()

    return decorated_function
  return decorator


def execute_python_function(
  code_files: dict[str, str],
  args: list,
  timeout: float,
  evaluation_script: str,
) -> str:
  """Execute a Python function in a temporary directory."""
  # Restore the original repository state using rsync
  subprocess.run(["rsync", "-a", "--delete", BACKUP_DIR, REPO_DIR])

  args = [f'"{arg}"' if isinstance(arg, str) else f"{arg}" for arg in args]

  for rel_path, range_and_content in code_files.items():
    with open(os.path.join(REPO_DIR, rel_path), "w", encoding="utf-8") as f:
      f.write(range_and_content)

  # Run the Python program
  # Use system python3 instead of sys.executable to avoid issues with PyInstaller bundles
  # TODO: if people write evaluation_script with a different language extension,
  # we should detect that and choose the right executable (e.g. node for .js)
  python_executable = os.environ.get("PYTHON_EXECUTABLE", "python3")
  try:
    output = common_tools.run_command(
      [python_executable, evaluation_script] + args, REPO_DIR, timeout
    )
    return output
  except common_tools.FunctionExecutionError as e:
    logger.info(
      "Run command failed: %s. Attempting to read checkpoint data.", e
    )
    try:
      # If the script leaves checkpointed json data, find and return it
      output = common_tools.run_command(["cat", "checkpoint.json"], REPO_DIR)
      return jsonify({"status": "success", "data": {"output": output, "metainfo": "Checkpoint"}}), 200
    except common_tools.FunctionExecutionError as ee:
      logger.info(
        "Failed to read checkpoint data: %s. Returning original error.", ee
      )
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

    result = execute_python_function(
      code, args, timeout, evaluation_script
    )
    return result, 200

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


def _git(args: list[str]) -> list[str]:
  """
  Wrapper around git which uses an external git directory to track changes
  without affecting the actual repo.
  """
  return ["git", "--git-dir", TRACKING_GIT_DIR, "--work-tree", REPO_DIR] + args


def init_git_tracking():
  """Initialize an external git tracking repo to capture workspace changes."""
  tracking = Path(TRACKING_GIT_DIR)
  if tracking.exists():
    shutil.rmtree(tracking)

  commands = [
    _git(["init"]),
    _git(["config", "user.email", "agent@docker"]),
    _git(["config", "user.name", "Agent"]),
    _git(["add", "-A"]),
    _git(["commit", "-m", "Initial state before agent run", "--allow-empty"]),
  ]

  for command in commands:
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
    )
    if result.returncode != 0:
      raise RuntimeError(
        f"Failed to initialize git tracking with command {command}: "
        f"{result.stderr.strip() or result.stdout.strip()}"
      )


def get_changed_files():
  """Return a dict of {relative_path: content} for all files changed since init."""
  EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf899d15363d7d95d"
  initial_commit = subprocess.run(
    _git(["rev-list", "--max-parents=0", "HEAD"]),
    capture_output=True,
    text=True,
  )
  if initial_commit.returncode != 0 or not initial_commit.stdout.strip():
    initial_sha = EMPTY_TREE_SHA
  else:
    initial_sha = initial_commit.stdout.strip().split('\n')[0]

  subprocess.run(_git(["add", "-A"]), capture_output=True)
  result = subprocess.run(
    _git(["diff", "--staged", "--name-only", initial_sha]),
    capture_output=True,
    text=True,
  )
  files = {}
  for name in result.stdout.strip().splitlines():
    path = os.path.join(REPO_DIR, name)
    if os.path.exists(path):
      try:
        with open(path, "r", encoding="utf-8") as f:
          files[name] = f.read()
      except (UnicodeDecodeError, ValueError):
        pass
    else:
      files[name] = None
  return files


def execute_shell_command(
  cmd: str, cwd: str, code_files: dict[str, str], timeout: float,
) -> "tuple[str, dict[str, Optional[str]]]":
  """Execute a shell command.

  Returns the shell command output to give back to the LLM, and a mapping of
  changed relative file paths to their current contents (None for deletions).
  """
  # Restore the original repository state using rsync
  subprocess.run(["rsync", "-a", "--delete", BACKUP_DIR, REPO_DIR])

  for rel_path, content in code_files.items():
    full_path = (Path(REPO_DIR) / rel_path).resolve()
    if not full_path.is_relative_to(Path(REPO_DIR).resolve()):
      raise ValueError(f"Path escapes repo directory: {rel_path}")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")

  init_git_tracking()

  # Run the shell command
  try:
    result = subprocess.run(
      cmd,
      shell=True,
      cwd=cwd,
      capture_output=True,
      encoding="utf-8",
      errors="replace",
      timeout=timeout,
    )
    parts = []
    if result.stdout:
      parts.append(result.stdout)
    if result.stderr:
      parts.append(f"STDERR:\n{result.stderr}")
    if result.returncode != 0:
      parts.append(f"Exit code: {result.returncode}")
    output = "\n".join(parts) if parts else "(no output)"
  except subprocess.TimeoutExpired:
    output = f"Error: command timed out after {timeout}s"

  return output, get_changed_files()


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
    cwd = (Path(REPO_DIR) / data["cwd"]).resolve()
    if not cwd.is_relative_to(Path(REPO_DIR).resolve()):
      return jsonify(
        {"status": "failed", "error": "cwd escapes repo directory"}
      ), 400
    code_files = data["code_files"]
    timeout = float(data.get("timeout", 120))

    logger.info("Executing shell command with timeout=%s", timeout)

    output, files = execute_shell_command(cmd, str(cwd), code_files, timeout)

    return jsonify({"status": "success", "result": {"output": output, "files": files}}), 200

  except Exception as e:
    logger.error("Agent execution failed: %s", e)
    return jsonify({"status": "failed", "error": str(e)}), 500


if __name__ == "__main__":
  # Ensure required directories exist
  os.makedirs(REPO_DIR, exist_ok=True)
  os.makedirs(BACKUP_DIR, exist_ok=True)

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

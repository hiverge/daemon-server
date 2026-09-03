"""Daemon server that dials in to the coordinator and executes its commands.

The relationship with the coordinator is inverted from the original design: the
daemon no longer serves inbound HTTP. Instead it opens an outbound connection to
the coordinator and stays connected, the coordinator pushes commands over
server-sent events (SSE), and the daemon POSTs each result back keyed by the
command's ``request_id``.

Authentication uses a Keycloak client-credentials token (client id / secret from
env vars). Tyk validates the token and injects identity headers, so the daemon
only needs to present a Bearer token.

The blocking executor functions (``execute_python_function``,
``execute_shell_command`` and the git-tracking helpers) are unchanged from the
inbound-serving design; they run in worker threads so the asyncio event loop
driving the SSE stream is never blocked.
"""

# Needed for Python 3.8 type hint compatibility
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

import aiohttp
import common_tools
from flask import Flask, jsonify
from pythonjsonlogger import json as json_logger

REPO_DIR = os.environ.get("REPO_DIR", "/app/")  # Directory where the repository is mounted
BACKUP_DIR = os.environ.get("BACKUP_DIR", "/shared/repo/")  # Backup directory to restore original state
TRACKING_GIT_DIR = os.environ.get("TRACKING_GIT_DIR", "/tmp/.agent_git")  # Git directory for tracking changes without affecting the actual repo

app = Flask(__name__)
sandbox_lock = threading.Lock()

# Token cache lifetime. We also refresh reactively on a 401, so this only bounds
# how long a still-valid token is reused.
TOKEN_REFRESH_PERIOD = 30 * 60  # 30 minutes

# Reconnect backoff bounds (seconds) for the outer SSE loop.
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0

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


# --------------------------------------------------------------------------- #
# Blocking executor functions (unchanged behavior, run in worker threads).
# --------------------------------------------------------------------------- #
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
      return json.dumps(
        {
          "status": "success",
          "result": {"output": output, "metainfo": "Checkpoint"},
        }
      )
    except common_tools.FunctionExecutionError as ee:
      logger.info(
        "Failed to read checkpoint data: %s. Returning original error.", ee
      )
      # Re-raise the original error unchanged. `run_command` already produces
      # the final, user-facing message (including any "Execution failed:"
      # prefix), so it must not be wrapped again here.
      raise e


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


# --------------------------------------------------------------------------- #
# Command execution: turn a coordinator command into a response dict.
# --------------------------------------------------------------------------- #
# A router name maps to the async callable that executes it. The runners are
# built once and passed explicitly to run_command_locally so they can be
# substituted with fakes in tests (dependency injection rather than patching).
Runner = Callable[[dict], Awaitable[dict]]


def parse_run_code_result(raw_output: str) -> dict:
  """Parses the evaluator's stdout into a coordinator response dict.

  The evaluator prints a JSON object on the final stdout line, already in the
  ``{status, result, error}`` form the coordinator expects, so a successful run
  is simply that parsed object. Anything that is not valid JSON is reported as a
  format failure.
  """
  try:
    return json.loads(raw_output)
  except (json.JSONDecodeError, TypeError) as e:
    return {
      "status": "failed",
      "error": f"Evaluator output is not valid JSON: {e}. Got: {raw_output}",
    }


async def _run_code_command(payload: dict) -> dict:
  """Runs a run_code command and returns a response dict."""
  code = payload.get("code")
  timeout = float(payload.get("timeout"))
  args = payload.get("args", ())
  evaluation_script = payload.get("evaluation_script", "evaluator.py")

  logger.info(
    "Executing code with timeout=%s, evaluation_script=%s",
    timeout,
    evaluation_script,
  )
  try:
    raw_output = await asyncio.to_thread(
      execute_python_function, code, args, timeout, evaluation_script
    )
    return parse_run_code_result(raw_output)
  except common_tools.FunctionExecutionError as e:
    return {"status": "failed", "error": str(e)}
  except subprocess.SubprocessError as e:
    if str(e) == "Exception occurred in preexec_fn.":
      return {
        "status": "failed",
        "error": "Execution failed: Memory limit exceeded",
      }
    return {"status": "failed", "error": "Internal server error"}


async def _run_shell_command(payload: dict) -> dict:
  """Runs a shell command and returns a response dict."""
  missing = [f for f in ("cmd", "cwd", "code_files") if f not in payload]
  if missing:
    return {
      "status": "failed",
      "error": f"Missing required field(s): {', '.join(missing)}",
    }

  cwd = (Path(REPO_DIR) / payload["cwd"]).resolve()
  if not cwd.is_relative_to(Path(REPO_DIR).resolve()):
    return {"status": "failed", "error": "cwd escapes repo directory"}

  cmd = payload["cmd"]
  code_files = payload["code_files"]
  timeout = float(payload.get("timeout", 120))

  logger.info("Executing shell command with timeout=%s", timeout)
  try:
    output, files = await asyncio.to_thread(
      execute_shell_command, cmd, str(cwd), code_files, timeout
    )
    return {"status": "success", "result": {"output": output, "files": files}}
  except Exception as e:
    return {"status": "failed", "error": str(e)}


async def _run_agent_command(payload: dict) -> dict:
  """Stub for the unimplemented agent command.

  Returns a clean failure so the coordinator's run_agent surfaces an error
  instead of hanging on a response that never arrives.
  """
  return {"status": "failed", "error": "agent not implemented"}


def build_runners() -> dict[str, Runner]:
  """Returns the mapping of router name to its execution callable."""
  return {
    "run_code": _run_code_command,
    "shell": _run_shell_command,
    "agent": _run_agent_command,
  }


async def run_command_locally(
  router: str, payload: dict, runners: dict[str, Runner]
) -> dict:
  """Executes a command locally, enforcing one evaluation at a time.

  Returns ``{"status": "busy"}`` when an evaluation is already running (the
  coordinator treats this as retryable), a router-specific response dict
  otherwise.
  """
  if not sandbox_lock.acquire(blocking=False):
    logger.info("Evaluation already running. Reporting busy.")
    return {"status": "busy"}
  try:
    runner = runners.get(router)
    if runner is None:
      return {"status": "failed", "error": f"Unknown router: {router}"}
    return await runner(payload)
  finally:
    sandbox_lock.release()


# --------------------------------------------------------------------------- #
# Keycloak client-credentials token handling.
# --------------------------------------------------------------------------- #
async def get_token(
  session: aiohttp.ClientSession,
  keycloak_url: str,
  realm: str,
  client_id: str,
  client_secret: str,
) -> str:
  """Fetches a client-credentials access token from Keycloak."""
  url = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/token"
  auth = aiohttp.BasicAuth(client_id, client_secret)
  data = {"grant_type": "client_credentials"}
  async with session.post(url, data=data, auth=auth) as response:
    response.raise_for_status()
    body = await response.json()
  return body["access_token"]


class TokenProvider:
  """Caches a Keycloak access token and refreshes it lazily or on demand."""

  def __init__(
    self,
    session: aiohttp.ClientSession,
    keycloak_url: str,
    realm: str,
    client_id: str,
    client_secret: str,
  ):
    self._session = session
    self._keycloak_url = keycloak_url
    self._realm = realm
    self._client_id = client_id
    self._client_secret = client_secret
    self._token: str | None = None
    self._fetched_at = 0.0

  async def get(self, force_refresh: bool) -> str:
    """Returns a cached token, fetching a fresh one when stale or forced."""
    now = time.monotonic()
    is_stale = (
      self._token is None or (now - self._fetched_at) > TOKEN_REFRESH_PERIOD
    )
    if force_refresh or is_stale:
      self._token = await get_token(
        self._session,
        self._keycloak_url,
        self._realm,
        self._client_id,
        self._client_secret,
      )
      self._fetched_at = now
    return self._token


# --------------------------------------------------------------------------- #
# SSE client: subscribe, dispatch commands, POST responses.
# --------------------------------------------------------------------------- #
async def handle_command(
  session: aiohttp.ClientSession,
  coordinator_url: str,
  token_provider: TokenProvider,
  cmd: dict,
  runners: dict[str, Runner],
) -> None:
  """Executes a single command and POSTs its result back to the coordinator."""
  request_id = cmd["request_id"]
  router = cmd["router"]
  payload = cmd.get("payload", {})

  result = await run_command_locally(router, payload, runners)
  body = {"request_id": request_id, "response": result}
  url = f"{coordinator_url}/daemon/response"

  # Post the result, refreshing the token and retrying once on a 401.
  for attempt in range(2):
    token = await token_provider.get(force_refresh=attempt > 0)
    headers = {"Authorization": f"Bearer {token}"}
    async with session.post(url, json=body, headers=headers) as response:
      if response.status == 401 and attempt == 0:
        logger.info("Response POST got 401; refreshing token and retrying.")
        continue
      if response.status >= 400:
        logger.error(
          "Failed to POST response for %s: HTTP %s",
          request_id,
          response.status,
        )
      return


async def subscribe_loop(
  session: aiohttp.ClientSession,
  coordinator_url: str,
  token_provider: TokenProvider,
  runners: dict[str, Runner],
) -> None:
  """Opens the SSE stream and dispatches each command it delivers.

  Commands are handled one at a time (the sandbox lock enforces this anyway),
  which keeps a single daemon from over-committing. Returns when the stream
  ends; the caller's reconnect loop reopens it.
  """
  token = await token_provider.get(force_refresh=False)
  headers = {"Authorization": f"Bearer {token}", "Accept": "text/event-stream"}
  url = f"{coordinator_url}/daemon/subscribe"

  async with session.get(url, headers=headers) as response:
    response.raise_for_status()
    logger.info("Subscribed to coordinator at %s", url)
    async for raw_line in response.content:
      line = raw_line.decode("utf-8").strip()
      # SSE comment lines (heartbeats) start with ':' and are ignored.
      if not line or line.startswith(":"):
        continue
      if not line.startswith("data:"):
        continue
      data = line[len("data:"):].strip()
      try:
        event = json.loads(data)
      except json.JSONDecodeError:
        logger.error("Ignoring malformed SSE data line: %s", data)
        continue
      if event.get("type") == "command":
        await handle_command(
          session, coordinator_url, token_provider, event, runners
        )
      else:
        logger.info("Ignoring non-command event: %s", event.get("type"))


async def run_forever(
  subscribe_fn: Callable[[], Awaitable[None]],
  sleep_fn: Callable[[float], Awaitable[None]],
  max_reconnects: int | None = None,
) -> None:
  """Runs ``subscribe_fn`` in a reconnect loop with exponential backoff + jitter.

  Backoff resets after a clean return so a long-lived connection that eventually
  drops reconnects promptly. Jitter spreads reconnects so many daemons do not
  stampede the coordinator after a restart. ``max_reconnects`` bounds the loop
  for tests; production passes None to run forever.
  """
  attempts = 0
  backoff = INITIAL_BACKOFF
  while max_reconnects is None or attempts < max_reconnects:
    try:
      await subscribe_fn()
      backoff = INITIAL_BACKOFF
    except asyncio.CancelledError:
      raise
    except Exception as e:
      logger.error("SSE connection dropped: %s", e)
    attempts += 1
    delay = min(backoff, MAX_BACKOFF) * (1.0 + random.random())
    logger.info("Reconnecting in %.1fs", delay)
    await sleep_fn(delay)
    backoff = min(backoff * 2, MAX_BACKOFF)


def _require_env(name: str) -> str:
  """Returns the value of a required environment variable or raises."""
  value = os.environ.get(name)
  if not value:
    raise RuntimeError(f"Required environment variable {name} is not set")
  return value


async def main_async() -> None:
  """Builds the client and runs the reconnecting subscribe loop forever."""
  coordinator_url = _require_env("COORDINATOR__URL").rstrip("/")
  keycloak_url = _require_env("KEYCLOAK_URL").rstrip("/")
  realm = _require_env("KEYCLOAK_REALM")
  client_id = _require_env("DAEMON_CLIENT_ID")
  client_secret = _require_env("DAEMON_CLIENT_SECRET")

  runners = build_runners()

  # A high read timeout: the SSE stream is long-lived and kept alive by the
  # coordinator's heartbeats. Command results return on separate POSTs, so a
  # long-running evaluation never holds the stream open waiting on a reply.
  timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
  async with aiohttp.ClientSession(timeout=timeout) as session:
    token_provider = TokenProvider(
      session, keycloak_url, realm, client_id, client_secret
    )

    async def subscribe_once() -> None:
      await subscribe_loop(session, coordinator_url, token_provider, runners)

    await run_forever(subscribe_once, asyncio.sleep)


@app.route("/health", methods=["GET"])
def health_check():
  """Health check endpoint served on a background thread for probes."""
  return jsonify({"status": "healthy"}), 200


def _serve_health(port: int) -> None:
  """Serves the /health endpoint using waitress (blocking; run in a thread)."""
  try:
    from waitress import serve

    serve(app, host="0.0.0.0", port=port)
  except ImportError:
    logger.warning("waitress not found, using Flask development server")
    app.run(debug=False, host="0.0.0.0", port=port)


if __name__ == "__main__":
  # Ensure required directories exist
  os.makedirs(REPO_DIR, exist_ok=True)
  os.makedirs(BACKUP_DIR, exist_ok=True)

  # Serve /health on a background thread for k8s probes during the transition.
  health_port = int(os.environ.get("PORT", "8080"))
  threading.Thread(
    target=_serve_health, args=(health_port,), daemon=True
  ).start()
  logger.info("Health server started on port %s", health_port)

  asyncio.run(main_async())

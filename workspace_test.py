"""Exercise workspace mode through the same Flask app as evaluation mode."""

import os
import shlex
import threading
import venv

import pytest

import common_tools
import main
from workspace import Workspace


@pytest.fixture
def runtime(tmp_path, monkeypatch):
  """Provide an isolated repository and real Bash runtime through Flask.

  Suppress only the server's scheduled process exit so failure/reset paths can
  be asserted in pytest. The separate image test exercises actual container exit.
  """
  repo, backup = tmp_path / "repo", tmp_path / "backup"
  backup.mkdir()
  (backup / "code.py").write_text("original\n")
  (backup / "remove.py").write_text("remove me\n")
  monkeypatch.setattr(main, "REPO_DIR", str(repo))
  monkeypatch.setattr(main, "BACKUP_DIR", str(backup))
  workspace = Workspace(
    str(repo), str(backup), main._resolve_repo_path, main.apply_code_files
  )
  monkeypatch.setattr(main, "workspace", workspace)
  # A unit test must not exit the pytest process. Container exit is tested
  # separately with the actual image.
  monkeypatch.setattr(
    main.threading,
    "Timer",
    lambda *args: type("NoExit", (), {"start": lambda self: None})(),
  )
  common_tools._contaminated.clear()
  yield workspace, main.app.test_client()
  workspace.close()
  common_tools._contaminated.clear()


HEADERS = {
  "X-Workspace-Session-ID": "session-1",
}


def call(runtime, operation, data=None, **headers):
  """Issue a workspace request, allowing session-header overrides for failures."""
  _, client = runtime
  path = "/shell" if operation == "shell" else f"/workspace/{operation}"
  return client.post(path, json=data or {}, headers=HEADERS | headers)


def result(runtime, operation, data=None):
  """Require a successful request and unwrap the daemon's result envelope."""
  response = call(runtime, operation, data)
  assert response.status_code == 200, response.json
  return response.json["result"]


def test_commands_files_and_terminal_share_one_workspace(runtime):
  ws, _ = runtime
  result(runtime, "start", {"files": {"code.py": "sampled\n", "remove.py": None}})
  result(
    runtime,
    "shell",
    {
      "cmd": (
        "mkdir sub\ncd sub\nexport ANSWER=42\n"
        "helper() { printf 'kept function'; }\nprintf 'created' > created.txt"
      )
    },
  )
  output = result(runtime, "shell", {"cmd": "printf '%s\\n' \"$ANSWER\"; pwd; helper"})[
    "output"
  ]
  assert "42" in output
  assert str(ws.repo / "sub") in output
  assert "kept function" in output
  assert result(runtime, "read", {"path": "sub/created.txt"}) == {"content": "created"}
  assert result(runtime, "snapshot")["files"] == {
    "code.py": "sampled\n",
    "sub/created.txt": "created",
  }
  assert result(runtime, "shell", {"cmd": "printf err >&2; false"})["output"].endswith(
    "Exit code: 1"
  )
  result(runtime, "apply", {"files": {"sub/created.txt": None, "new.py": "written"}})
  assert "created.txt" not in result(runtime, "list", {"path": "sub"})["entries"]
  assert (
    result(runtime, "shell", {"cmd": "cat new.py", "cwd": "."})["output"] == "written"
  )


def test_checkpoint_restores_files_but_not_terminal(runtime):
  ws, _ = runtime
  result(runtime, "start")
  result(runtime, "shell", {"cmd": "printf '\\377' > binary; export STATE=before"})
  checkpoint = result(runtime, "checkpoint")["checkpoint"]
  result(
    runtime,
    "shell",
    {
      "cmd": (
        "export STATE=after; rm code.py binary; printf new > extra; git config user.name Agent"
      )
    },
  )
  result(runtime, "restore", {"checkpoint": checkpoint})
  assert (ws.repo / "binary").read_bytes() == b"\xff"
  assert (ws.repo / "code.py").read_text() == "original\n"
  assert not (ws.repo / "extra").exists()
  output = result(runtime, "shell", {"cmd": 'echo "$STATE"; git config user.name'})[
    "output"
  ]
  assert output == "after\nAgent\n"
  assert ".git/config" not in result(runtime, "snapshot")["files"]


def test_virtualenv_activation_preserves_session_and_exit_status(runtime, tmp_path):
  """Prompt decoration from a real activation script must not break shell results."""
  ws, client = runtime
  environment = tmp_path / "venv"
  venv.EnvBuilder(with_pip=False).create(environment)
  result(runtime, "start")
  result(
    runtime,
    "shell",
    {
      "cmd": (
        "unset VIRTUAL_ENV_DISABLE_PROMPT\n"
        f"source {shlex.quote(str(environment / 'bin/activate'))}"
      )
    },
  )
  assert client.get("/health").status_code == 200
  assert not ws.retired
  output = result(
    runtime, "shell", {"cmd": 'python -c "import sys; print(sys.prefix)"'}
  )["output"]
  assert str(environment) in output
  assert result(runtime, "shell", {"cmd": "false"})["output"].endswith("Exit code: 1")
  result(runtime, "shell", {"cmd": "deactivate"})
  assert (
    result(runtime, "shell", {"cmd": 'printf "%s" "${VIRTUAL_ENV-unset}"'})["output"]
    == "unset"
  )


@pytest.mark.parametrize(
  "operation,data,expected_status",
  [
    ("snapshot", {}, 200),
    ("read", {"path": "pipe"}, 400),
    ("read", {"path": "pipe", "binary": True}, 400),
    ("apply", {"files": {"code.py": "unexpected", "pipe": "content"}}, 400),
    ("shell", {"cmd": "touch unexpected", "code_files": {"pipe": "content"}}, 400),
  ],
)
def test_file_operations_do_not_block_on_fifo(
  runtime, operation, data, expected_status
):
  """Skip FIFOs in snapshots and reject reads/writes without trapping the lock."""
  ws, _ = runtime
  result(runtime, "start")
  result(runtime, "shell", {"cmd": "mkfifo pipe"})
  responses = []
  thread = threading.Thread(
    target=lambda: responses.append(call(runtime, operation, data)), daemon=True
  )
  thread.start()
  try:
    thread.join(2)
    assert not thread.is_alive(), f"{operation} blocked while opening a FIFO"
    assert responses[0].status_code == expected_status
    if operation == "snapshot":
      assert responses[0].json["result"]["files"] == {
        "code.py": "original\n",
        "remove.py": "remove me\n",
      }
    assert (ws.repo / "code.py").read_text() == "original\n"
    assert not (ws.repo / "unexpected").exists()
    # Deleting the FIFO and creating an ordinary file remain valid operations.
    result(runtime, "apply", {"files": {"pipe": None, "new.txt": ""}})
    assert not (ws.repo / "pipe").exists()
    assert result(runtime, "read", {"path": "new.txt"}) == {"content": ""}
    assert result(runtime, "reset") == {"restart_required": True}
  finally:
    if thread.is_alive():
      # Unblock a regressed reader/writer so it releases the lock on failure.
      writing = operation in ("apply", "shell")
      mode = os.O_RDONLY if writing else os.O_WRONLY
      peer = os.open(ws.repo / "pipe", mode | os.O_NONBLOCK)
      if not writing:
        os.close(peer)
      thread.join(5)
      if writing:
        os.close(peer)


@pytest.mark.parametrize(
  "policy",
  [
    {"read_only": True},
    {"protected_files": ["code.py"]},
    {"editable_files": ["remove.py", "new.py"]},
  ],
)
def test_file_and_shell_writes_respect_session_policy(runtime, policy):
  ws, _ = runtime
  result(runtime, "start", policy)
  assert call(runtime, "apply", {"files": {"code.py": "bad"}}).status_code == 403
  result(
    runtime,
    "shell",
    {"cmd": "rm code.py; printf changed > remove.py; printf new > new.py"},
  )
  assert (ws.repo / "code.py").read_text() == "original\n"
  if policy.get("read_only"):
    assert (ws.repo / "remove.py").read_text() == "remove me\n"
    assert not (ws.repo / "new.py").exists()
  else:
    assert (ws.repo / "remove.py").read_text() == "changed"


def test_read_only_session_can_start_from_sampled_candidate(runtime):
  result(runtime, "start", {"read_only": True, "files": {"code.py": "sampled"}})
  assert result(runtime, "read", {"path": "code.py"})["content"] == "sampled"


def test_containment_git_exclusion_and_symlink_deletion(runtime, tmp_path):
  ws, _ = runtime
  result(runtime, "start")
  (tmp_path / "outside").write_text("outside")
  (ws.repo / "escape").symlink_to(tmp_path / "outside")
  (ws.repo / "alias").symlink_to("code.py")
  for name in ("../outside", "escape", ".git/config"):
    assert call(runtime, "read", {"path": name}).status_code == 400
    assert call(runtime, "apply", {"files": {name: "bad"}}).status_code == 400
  result(runtime, "apply", {"files": {"escape": None, "alias": None}})
  assert (tmp_path / "outside").read_text() == "outside"
  assert (ws.repo / "code.py").read_text() == "original\n"
  assert not (ws.repo / "alias").is_symlink()


def test_session_assignment_rejects_stale_requests(runtime):
  _, client = runtime
  assert client.post("/workspace/start", json={}).status_code == 400
  result(runtime, "start")
  assert call(runtime, "start").status_code == 409
  assert (
    call(runtime, "snapshot", **{"X-Workspace-Session-ID": "session-2"}).status_code
    == 403
  )
  assert (
    call(
      runtime, "shell", {"cmd": "true"}, **{"X-Workspace-Session-ID": "session-2"}
    ).status_code
    == 403
  )


def test_commands_hold_the_shared_lock_against_snapshots(runtime):
  ws, _ = runtime
  result(runtime, "start")
  started = threading.Event()
  responses = []

  def command():
    started.set()
    responses.append(call(runtime, "shell", {"cmd": "touch running; sleep 0.5"}))

  thread = threading.Thread(target=command)
  thread.start()
  started.wait(2)
  # Wait for actual command execution, not just the request thread starting.
  import time

  deadline = time.monotonic() + 5
  while not (ws.repo / "running").exists() and time.monotonic() < deadline:
    time.sleep(0.01)
  assert (ws.repo / "running").exists()
  assert call(runtime, "snapshot").status_code == 429
  thread.join(5)
  assert responses[0].status_code == 200


def test_background_process_is_stopped_before_next_operation(runtime):
  ws, _ = runtime
  result(runtime, "start")
  result(runtime, "shell", {"cmd": "sleep 60 & echo $! > background.pid"})
  pid = int((ws.repo / "background.pid").read_text())
  import psutil

  assert (
    not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
  )


@pytest.mark.parametrize("command", ["sleep 60", "while :; do :; done"])
def test_timeout_interrupts_command_and_preserves_session(runtime, command):
  """Interrupt external commands and Bash loops without losing session state."""
  ws, client = runtime
  result(runtime, "start")
  shell_pid = ws.terminal.child.pid
  output = result(
    runtime,
    "shell",
    {
      "cmd": (
        "mkdir sub; cd sub; export ANSWER=42\n"
        "helper() { printf 'kept function'; }\n"
        "printf partial > partial.txt\n"
        "printf 'before timeout\\n'\n"
        f"{command}\n"
        "printf unexpected > after-timeout.txt"
      ),
      "timeout": 0.2,
    },
  )["output"]
  assert output.count("before timeout") == 1
  assert "Error: command timed out after 0.2s" in output
  assert not ws.retired
  assert ws.terminal.child.pid == shell_pid
  assert client.get("/health").status_code == 200
  assert (ws.repo / "sub/partial.txt").read_text() == "partial"
  assert not (ws.repo / "sub/after-timeout.txt").exists()
  output = result(
    runtime,
    "shell",
    {"cmd": 'echo "$ANSWER"; pwd; helper; printf next > next.txt'},
  )["output"]
  assert output == f"42\n{ws.repo / 'sub'}\nkept function"
  assert result(runtime, "read", {"path": "sub/next.txt"}) == {"content": "next"}
  assert result(runtime, "reset") == {"restart_required": True}


@pytest.mark.parametrize(
  "command", ["exit", "exec /bin/true", "trap '' INT; while :; do :; done"]
)
def test_terminal_loss_or_failed_interrupt_retires_workspace(runtime, command):
  """Retire only when Bash exits or cannot return to its prompt after Ctrl+C."""
  ws, client = runtime
  result(runtime, "start")
  assert call(runtime, "shell", {"cmd": command, "timeout": 0.2}).status_code == 409
  assert ws.retired
  assert client.get("/health").status_code == 503
  assert call(runtime, "snapshot").status_code == 409


def test_reset_requires_container_replacement(runtime):
  _, client = runtime
  result(runtime, "start")
  assert result(runtime, "reset") == {"restart_required": True}
  assert client.get("/health").status_code == 503
  assert (
    call(runtime, "start", **{"X-Workspace-Session-ID": "session-2"}).status_code == 409
  )
  assert call(runtime, "shell", {"cmd": "true"}).status_code == 409


def test_workspace_rejects_evaluation_requests(runtime):
  _, client = runtime
  response = client.post("/run_code", json={}, headers=HEADERS)
  assert response.status_code == 404

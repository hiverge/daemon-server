"""Focused unit tests for workspace assignments and request handling."""

import threading
from unittest.mock import Mock

import pytest

import common_tools
import main
from workspace import WorkspaceError, WorkspaceRuntime


@pytest.fixture
def runtime(tmp_path, monkeypatch):
  baseline, storage = tmp_path / "baseline", tmp_path / "storage"
  baseline.mkdir()
  storage.mkdir()
  runtime = WorkspaceRuntime(baseline, storage)
  runtime.ready = True
  monkeypatch.setattr(runtime, "_admin_command", Mock())
  monkeypatch.setattr(main, "workspace_runtime", runtime)
  monkeypatch.setattr(main, "DAEMON_MODE", "workspace")
  monkeypatch.setattr(common_tools, "_contaminated", threading.Event())
  return runtime


def test_assignment_reuse_and_stale_requests(runtime):
  client = main.app.test_client()
  runtime.start("first")
  first_session = runtime.session_dir
  with pytest.raises(WorkspaceError, match="already assigned"):
    runtime.start("second")
  runtime.release("first")
  assert runtime.state == "idle"
  assert not first_session.exists()
  with pytest.raises(WorkspaceError, match="already been used"):
    runtime.start("first")

  runtime.start("second")
  for route in ("/workspace/health", "/workspace/release", "/shell"):
    response = client.post(route, json={"assignment_token": "first"})
    assert response.status_code == 409
  assert runtime.assignment_token == "second"
  assert runtime.session_dir.exists()


def test_shell_uses_workspace_and_existing_response_envelope(runtime, monkeypatch):
  runtime.start("test")
  execute = Mock(return_value=("hello\n", 0))
  monkeypatch.setattr(runtime, "_execute", execute)
  evaluation = Mock(side_effect=AssertionError("evaluation runner called"))
  monkeypatch.setattr(main, "execute_shell_command", evaluation)

  response = main.app.test_client().post(
    "/shell",
    json={
      "assignment_token": "test",
      "cmd": "echo hello",
      "cwd": "nested",
      "code_files": {},
      "timeout": 15,
    },
  )

  assert response.json == {
    "status": "success",
    "result": {"output": "hello\n", "files": {}},
  }
  execute.assert_called_once_with("echo hello", "/app/nested", 15.0)
  evaluation.assert_not_called()


@pytest.mark.parametrize(
  "extra",
  [
    {"assignment_token": None},
    {"cmd": []},
    {"cwd": "\0"},
    {"timeout": 0},
    {"timeout": "nan"},
    {"timeout": 10**400},
    {"code_files": {"a.py": "unsupported overlay"}},
  ],
)
def test_invalid_shell_request_preserves_assignment(runtime, extra):
  runtime.start("test")
  response = main.app.test_client().post(
    "/shell",
    json={
      "assignment_token": "test",
      "cmd": "true",
      **extra,
    },
  )
  assert response.status_code == 400
  assert runtime.assignment_token == "test"
  assert runtime.state == "active"


@pytest.mark.parametrize("body", ["{", "[]"])
def test_invalid_json_uses_error_envelope(runtime, body):
  response = main.app.test_client().post(
    "/workspace/start",
    data=body,
    content_type="application/json",
  )
  assert response.status_code == 400
  assert response.json["status"] == "failed"


@pytest.mark.parametrize("operation", ["start", "release"])
def test_filesystem_failure_retires_runtime(runtime, monkeypatch, operation):
  if operation == "release":
    runtime.start("test")
    monkeypatch.setattr("workspace.os.path.ismount", lambda _: True)
  runtime._admin_command.side_effect = OSError("injected mount failure")
  client = main.app.test_client()

  response = client.post(f"/workspace/{operation}", json={"assignment_token": "test"})

  assert response.status_code == 503
  assert runtime.state == "retired"
  assert client.get("/health").status_code == 503
  assert (
    client.post("/workspace/start", json={"assignment_token": "next"}).status_code
    == 503
  )


def test_workspace_cleanup_failure_does_not_schedule_restart(runtime, monkeypatch):
  runtime.retired = True
  common_tools.mark_contaminated()
  timer = Mock()
  monkeypatch.setattr(main.threading, "Timer", timer)
  assert main.app.test_client().get("/health").status_code == 503
  timer.assert_not_called()


def test_session_operations_share_existing_lock(runtime):
  client = main.app.test_client()
  with main.sandbox_lock:
    for route in (
      "/workspace/start",
      "/workspace/health",
      "/workspace/release",
      "/shell",
    ):
      assert client.post(route, json={"assignment_token": "test"}).status_code == 429
    assert client.get("/health").status_code == 200


def test_mode_separation(runtime, monkeypatch):
  client = main.app.test_client()
  assert client.post("/run_code", json={}).status_code == 404
  monkeypatch.setattr(main, "DAEMON_MODE", "evaluation")
  assert client.get("/health").json == {"status": "healthy"}
  for route in ("/workspace/start", "/workspace/health", "/workspace/release"):
    assert client.post(route, json={}).status_code == 404

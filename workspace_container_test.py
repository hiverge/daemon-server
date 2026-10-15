"""Run with WORKSPACE_TEST_IMAGE pointing at a built Dockerfile.workspace image."""

import os
import shlex
import subprocess
import time

import pytest
import requests


pytestmark = pytest.mark.skipif(
  not os.getenv("WORKSPACE_TEST_IMAGE"), reason="requires workspace image"
)


@pytest.fixture
def workspace_container(tmp_path):
  """Provide real PID 1 daemons sharing only a read-only baseline, then remove them."""
  baseline = tmp_path / "baseline"
  baseline.mkdir()
  (baseline / "code.py").write_text("baseline")
  containers = []

  def docker(*args):
    """Run a Docker command and return its stdout, failing on command errors."""
    return subprocess.check_output(["docker", *args], text=True).strip()

  def start():
    """Launch a disposable workspace and wait until its HTTP server is ready."""
    container = docker(
      "run",
      "-d",
      "-p",
      "127.0.0.1::8080",
      "-v",
      f"{baseline}:/shared/repo:ro",
      os.environ["WORKSPACE_TEST_IMAGE"],
    )
    containers.append(container)
    port = docker("port", container, "8080/tcp").rsplit(":", 1)[1]
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
      try:
        if requests.get(f"{url}/health", timeout=1).status_code == 200:
          return container, url
      except requests.RequestException:
        pass
      time.sleep(0.1)
    pytest.fail(f"Workspace failed to start: {docker('logs', container)}")

  def call(url, session, operation, **data):
    """Send an HTTP request for the assigned session and require success."""
    route = "/shell" if operation == "shell" else f"/workspace/{operation}"
    response = requests.post(
      url + route, json=data, headers={"X-Workspace-Session-ID": session}, timeout=15
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]

  try:
    yield baseline, docker, start, call
  finally:
    for container in containers:
      subprocess.run(["docker", "rm", "-f", container], capture_output=True)


def test_container_release_and_clean_reuse(workspace_container):
  """Prove real session persistence and complete reset by container replacement.

  Both containers share only a read-only repository baseline. The second must
  inherit neither repository edits nor state created elsewhere in the first.
  """
  baseline, docker, start, call = workspace_container
  first, url = start()
  call(url, "first", "start")
  call(
    url,
    "first",
    "shell",
    cmd=(
      "mkdir sub; cd sub; export SESSION_VALUE=kept; printf local > local.txt; "
      'touch "$HOME/workspace-marker" /tmp/workspace-marker /usr/local/workspace-marker'
    ),
  )
  output = call(url, "first", "shell", cmd='echo "$SESSION_VALUE"; pwd; cat local.txt')[
    "output"
  ]
  assert output == "kept\n/app/sub\nlocal"
  checkpoint = call(url, "first", "checkpoint")["checkpoint"]
  call(url, "first", "shell", cmd="rm ../code.py; printf later > later.txt")
  call(url, "first", "restore", checkpoint=checkpoint)
  assert call(url, "first", "snapshot")["files"] == {
    "code.py": "baseline",
    "sub/local.txt": "local",
  }
  assert (
    call(url, "first", "shell", cmd='echo "$SESSION_VALUE"; pwd')["output"]
    == "kept\n/app/sub\n"
  )
  assert call(url, "first", "reset") == {"restart_required": True}
  subprocess.run(["docker", "wait", first], check=True, capture_output=True, timeout=10)
  assert docker("inspect", "-f", "{{.State.Running}}", first) == "false"

  # Kubernetes creates a new container filesystem on restart. Docker restart
  # retains its writable layer, so create a replacement here to match kubelet.
  _, url = start()
  call(url, "second", "start")
  assert call(url, "second", "snapshot")["files"] == {"code.py": "baseline"}
  output = call(
    url,
    "second",
    "shell",
    cmd=(
      'test -z "$SESSION_VALUE" && '
      'test ! -e "$HOME/workspace-marker" && '
      "test ! -e /tmp/workspace-marker && "
      "test ! -e /usr/local/workspace-marker && pwd"
    ),
  )["output"]
  assert output == "/app\n"
  assert (baseline / "code.py").read_text() == "baseline"


@pytest.mark.parametrize(
  "policy", [{"protected_files": ["code.py"]}, {"read_only": True}]
)
@pytest.mark.parametrize("times_out", [False, True])
def test_detached_writer_stops_before_shell_success(
  workspace_container, policy, times_out
):
  """Clean detached writers and restore policy after completion or interruption."""
  _, docker, start, call = workspace_container
  container, url = start()
  call(url, "session", "start", **policy)
  # The original parent waits for a marker from the detached grandchild. This
  # guarantees it has escaped Bash before /shell begins cleanup, without sleeps
  # that guess how quickly a loaded CI worker can run the command.
  script = """
import os
from pathlib import Path
import time

pid_file = Path("/tmp/detached.pid")
child = os.fork()
if child:
    os.waitpid(child, 0)
    while not pid_file.exists():
        time.sleep(0.01)
    os._exit(0)
os.setsid()
if os.fork():
    os._exit(0)
Path("/app/code.py").write_text("during command")
pid_file.write_text(str(os.getpid()))
while not Path("/tmp/release-writer").exists():
    time.sleep(0.01)
Path("/app/code.py").write_text("after shell success")
"""
  command = f"export SESSION_VALUE=kept\npython3 -c {shlex.quote(script)}"
  if times_out:
    command += "\nsleep 60"
  output = call(
    url,
    "session",
    "shell",
    cmd=command,
    timeout=1 if times_out else 120,
  )["output"]
  if times_out:
    assert "Error: command timed out after 1s" in output
  assert call(url, "session", "snapshot")["files"] == {"code.py": "baseline"}
  # Inspect from outside the daemon so another /shell cannot hide a leak by
  # cleaning it up. A live writer here could overwrite the restored candidate.
  docker(
    "exec",
    container,
    "/opt/daemon/.venv/bin/python",
    "-c",
    "from pathlib import Path; import psutil; "
    "pid = int(Path('/tmp/detached.pid').read_text()); "
    "assert not psutil.pid_exists(pid), 'Detached writer survived shell success'",
  )
  assert (
    call(url, "session", "shell", cmd='echo "$SESSION_VALUE"; pwd')["output"]
    == "kept\n/app\n"
  )
  assert call(url, "session", "reset") == {"restart_required": True}

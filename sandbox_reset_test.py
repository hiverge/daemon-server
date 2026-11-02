"""Recovery tests use synthetic roots; native process tests require a container."""

import errno
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import common_tools
import main
import sandbox_reset
from sandbox_reset import ResetIncomplete, SandboxReset


@pytest.fixture
def machine(tmp_path):
  root = tmp_path / "machine"
  for name in ("app", "etc", "tmp", "opt/env"):
    (root / name).mkdir(parents=True)
  (root / "etc/config").write_text("setup")
  (root / "opt/env/library").write_bytes(b"original library")
  (root / "etc/link").symlink_to("config")
  return root


@pytest.fixture
def controller(machine):
  controller = SandboxReset(
    machine, machine / "tmp/.reset", excluded=(machine / "app",), track_processes=False
  )
  controller.prepare()
  yield controller
  controller.close()


def drain(controller):
  for _ in range(1000):
    controller.maintain()
    if not controller.trash and not controller._partial and not controller._refill:
      return
    if controller.reason:
      return
  pytest.fail("maintenance did not finish")


def test_restores_baseline_and_preserves_app(controller, machine):
  timestamp = (machine / "etc").stat().st_mtime_ns
  controller.begin()
  (machine / "etc/config").write_text("damage")
  (machine / "etc/new").write_text("addition")
  (machine / "etc/link").unlink()
  (machine / "etc/link").symlink_to("../tmp/elsewhere")
  (machine / "opt/env/library").unlink()
  (machine / "app/result").write_text("keep")
  result = controller.finish()
  assert result.status == "clean", result
  assert (machine / "etc/config").read_text() == "setup"
  assert not (machine / "etc/new").exists()
  assert os.readlink(machine / "etc/link") == "config"
  assert (machine / "opt/env/library").read_bytes() == b"original library"
  assert (machine / "app/result").read_text() == "keep"
  assert (machine / "etc").stat().st_mtime_ns == timestamp


def test_in_place_edit_preserving_mtime_is_restored(controller, machine):
  path = machine / "etc/config"
  before = path.stat()
  controller.begin()
  path.write_text("other")
  os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
  assert controller.finish().status == "clean"
  assert path.read_text() == "setup"


def test_ancestor_symlink_cannot_redirect_restoration(controller, machine, tmp_path):
  outside = tmp_path / "untouched"
  outside.mkdir()
  (outside / "library").write_text("external")
  controller.begin()
  (machine / "opt/env").rename(machine / "tmp/moved")
  (machine / "opt/env").symlink_to(outside, target_is_directory=True)
  result = controller.finish()
  assert result.status in ("clean", "reclamation_pending"), result
  assert (machine / "opt/env/library").read_bytes() == b"original library"
  assert not (machine / "opt/env").is_symlink()
  assert (outside / "library").read_text() == "external"
  drain(controller)
  assert controller.available


def test_prepared_replacements_are_replenished(controller, machine):
  path = machine / "etc/config"
  files = controller.state_dir / "files"
  initial_count = len(list(files.iterdir()))
  initial_bytes = sum(file.stat().st_size for file in files.iterdir())
  for index in range(10):
    blob = controller.nodes[path].blob
    spare_inode = blob.spare.stat().st_ino
    controller.begin()
    path.write_text(f"overwrite {index}")
    assert controller.finish().status == "clean"
    assert path.read_text() == "setup"
    assert path.stat().st_ino == spare_inode
    drain(controller)
    assert blob.spare is not None
  assert len(list(files.iterdir())) == initial_count
  assert sum(file.stat().st_size for file in files.iterdir()) == initial_bytes


def test_hardlinks_restored_without_mutating_app_alias(machine):
  os.link(machine / "etc/config", machine / "etc/alias")
  controller = SandboxReset(
    machine, machine / "tmp/.reset", excluded=(machine / "app",), track_processes=False
  )
  try:
    controller.prepare()
    controller.begin()
    os.link(machine / "etc/config", machine / "app/alias")
    (machine / "etc/config").write_text("command result")
    assert controller.finish().status == "clean"
    assert (machine / "etc/config").read_text() == "setup"
    assert (machine / "etc/alias").read_text() == "setup"
    assert (machine / "etc/config").stat().st_ino == (
      machine / "etc/alias"
    ).stat().st_ino
    assert (machine / "app/alias").read_text() == "command result"
  finally:
    controller.close()


def test_quarantine_defers_tree_walk_and_blocks_reuse(controller, machine, monkeypatch):
  controller.begin()
  added = machine / "tmp/large"
  added.mkdir()
  for index in range(100):
    (added / str(index)).write_text("temporary")
  real_scan = sandbox_reset.os.scandir

  def scan(path):
    if isinstance(path, (str, Path)):
      assert "/trash/" not in str(path), "reset walked quarantined descendants"
      assert Path(path) != added, "reset walked the newly added tree"
    return real_scan(path)

  with monkeypatch.context() as patch:
    patch.setattr(sandbox_reset.os, "scandir", scan)
    result = controller.finish()
  assert result.status == "reclamation_pending", result
  assert not added.exists()
  assert not controller.available
  with pytest.raises(ResetIncomplete, match="reclamation"):
    controller.begin()
  drain(controller)
  assert controller.available
  assert not list((controller.state_dir / "trash").iterdir())


def test_quarantine_yields_between_filesystem_operations(controller, machine):
  controller.begin()
  (machine / "tmp/new").mkdir()
  (machine / "tmp/new/file").write_text("data")
  assert controller.finish().status == "reclamation_pending"
  controller.maintain()
  assert controller.trash
  assert not controller.available
  drain(controller)
  assert controller.reason is None
  assert controller.available


def test_sparse_file_beyond_former_archive_cap_is_restored(machine):
  path = machine / "etc/large"
  # Exercise archive, restoration and refill with a large sparse file.
  size = 65 * 1024**2
  with path.open("wb") as stream:
    stream.write(b"start")
    stream.seek(size - len(b"end"))
    stream.write(b"end")
  controller = SandboxReset(machine, machine / "tmp/.reset", track_processes=False)
  try:
    controller.prepare()
    blob = controller.nodes[path].blob
    assert isinstance(blob.archive, Path)
    assert blob.archive.stat().st_size == size
    controller.begin()
    path.write_bytes(b"damage")
    result = controller.finish()
    assert result.status == "clean", result
    assert path.stat().st_size == size
    with path.open("rb") as stream:
      assert stream.read(len(b"start")) == b"start"
      stream.seek(size - len(b"end"))
      assert stream.read() == b"end"
    drain(controller)
    assert blob.spare.stat().st_size == size
  finally:
    controller.close()


@pytest.mark.parametrize("stored_copy", ["spare", "archive"])
def test_modified_stored_copy_cannot_be_installed(controller, machine, stored_copy):
  path = machine / "etc/config"
  controller.begin()
  getattr(controller.nodes[path].blob, stored_copy).write_bytes(b"forged replacement")
  path.write_text("damage")
  result = controller.finish()
  assert result.status == "retire"
  assert "was modified" in result.reason
  assert path.read_text() != "forged replacement"


@pytest.mark.parametrize("escape", ["traversal", "symlink"])
def test_reset_storage_escape_is_rejected_before_writes(machine, tmp_path, escape):
  outside = tmp_path / "external"
  outside.mkdir()
  if escape == "symlink":
    (machine / "escape").symlink_to(outside, target_is_directory=True)
    state = machine / "escape/state"
  else:
    state = machine / "../external/state"
  with pytest.raises(ValueError, match="within the managed root"):
    SandboxReset(machine, state, track_processes=False)
  assert not list(outside.iterdir())


def test_notification_fallback_verifies_without_repo_scan(controller, machine):
  controller.notifications.close()
  controller.notifications.complete = False
  for index in range(1000):
    (machine / "app" / str(index)).write_text("source")
  assert not any(path.is_relative_to(machine / "app") for path in controller.nodes)
  controller.begin()
  (machine / "etc/config").write_text("changed")
  result = controller.finish()
  assert result.status == "clean", result
  assert (machine / "etc/config").read_text() == "setup"


def test_copy_refill_yields_and_resumes(controller, machine):
  path = machine / "etc/config"
  controller.begin()
  path.write_text("changed")
  assert controller.finish().status == "clean"
  controller.maintain()
  assert controller._partial is not None
  assert controller.nodes[path].blob.spare is None
  drain(controller)
  assert controller.nodes[path].blob.spare is not None
  assert controller.reason is None


def test_shared_memory_is_managed_while_device_tree_is_excluded(machine):
  shared = machine / "dev/shm"
  shared.mkdir(parents=True)
  (machine / "dev/device").write_text("leave alone")
  (shared / "setup").write_text("initial")
  controller = SandboxReset(
    machine,
    machine / "tmp/.reset",
    excluded=(machine / "dev",),
    additional_roots=(shared,),
    track_processes=False,
  )
  try:
    controller.prepare()
    controller.begin()
    (shared / "setup").write_text("damage")
    (shared / "new").write_text("addition")
    assert controller.finish().status == "clean"
    assert (shared / "setup").read_text() == "initial"
    assert not (shared / "new").exists()
    assert (machine / "dev/device").read_text() == "leave alone"
  finally:
    controller.close()


def _fake_process(proc, pid, tick, uid):
  directory = proc / str(pid)
  directory.mkdir(exist_ok=True)
  fields = ["S"] + ["0"] * 18 + [str(tick)]
  (directory / "stat").write_text(f"{pid} (worker with spaces) " + " ".join(fields))
  (directory / "status").write_text(f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\n")


def test_pid_reuse_detached_survivors_and_uid_scope(tmp_path, monkeypatch):
  proc = tmp_path / "proc"
  (proc / "sys/kernel").mkdir(parents=True)
  counter = proc / "sys/kernel/ns_last_pid"
  counter.write_text("900001")
  _fake_process(proc, 900001, 1, 123)
  tracker = sandbox_reset._Processes(uid=123, proc=proc)
  tracker.begin()
  _fake_process(proc, 900001, 2, 123)  # Reused PID with a new start time.
  _fake_process(proc, 900002, 3, 999)  # Different execution scope.
  kills = []

  def kill(pid, sig):
    kills.append(pid)
    if len(kills) == 2:
      shutil.rmtree(proc / str(pid))

  def no_pidfd(pid):
    raise OSError(errno.ENOSYS, "unsupported")

  monkeypatch.setattr(sandbox_reset.os, "kill", kill)
  monkeypatch.setattr(sandbox_reset.os, "pidfd_open", no_pidfd, raising=False)
  with pytest.raises(ResetIncomplete, match="survived cleanup"):
    tracker.stop(sandbox_reset._Work())
  assert tracker.suspects == {900001}
  tracker.stop(sandbox_reset._Work())
  assert kills == [900001, 900001]
  assert (proc / "900002").exists()
  assert not tracker.suspects


def test_pid_counter_unavailable_uses_full_discovery(tmp_path, monkeypatch):
  proc = tmp_path / "proc"
  proc.mkdir()
  _fake_process(proc, 900001, 1, 123)
  tracker = sandbox_reset._Processes(uid=123, proc=proc)
  tracker.begin()
  _fake_process(proc, 900002, 2, 123)
  kills = []

  def kill(pid, sig):
    kills.append(pid)
    shutil.rmtree(proc / str(pid))

  monkeypatch.setattr(sandbox_reset.os, "kill", kill)
  monkeypatch.setattr(
    sandbox_reset.os,
    "pidfd_open",
    lambda pid: (_ for _ in ()).throw(OSError(errno.ENOSYS, "unsupported")),
    raising=False,
  )
  tracker.stop(sandbox_reset._Work())
  assert kills == [900002]
  assert (proc / "900001").exists()


def test_root_execution_detects_descendants_that_changed_uid(tmp_path, monkeypatch):
  proc = tmp_path / "proc"
  proc.mkdir()
  _fake_process(proc, 900001, 1, 999)  # Legitimate non-root setup service.
  tracker = sandbox_reset._Processes(uid=0, proc=proc)
  tracker.begin()
  _fake_process(proc, 900002, 2, 999)  # New process dropped every root UID.
  killed = []

  def kill(pid, sig):
    killed.append(pid)
    shutil.rmtree(proc / str(pid))

  monkeypatch.setattr(sandbox_reset.os, "kill", kill)
  monkeypatch.setattr(
    sandbox_reset.os,
    "pidfd_open",
    lambda pid: (_ for _ in ()).throw(OSError(errno.ENOSYS, "unsupported")),
    raising=False,
  )
  tracker.stop(sandbox_reset._Work())
  assert killed == [900002]
  assert (proc / "900001").exists()


@pytest.fixture
def api(controller, machine, monkeypatch):
  common_tools._contaminated.clear()
  main._restart_scheduled.clear()
  monkeypatch.setattr(main, "reset_controller", controller)
  monkeypatch.setattr(main, "REPO_DIR", str(machine / "app"))
  monkeypatch.setattr(main, "SCRATCH_DIR", str(machine / "tmp/scratch"))
  monkeypatch.setattr(main, "TRACKING_GIT_DIR", str(controller.state_dir / "git"))
  monkeypatch.setattr(
    main, "SCRATCH_TRACKING_GIT_DIR", str(controller.state_dir / "scratch-git")
  )
  monkeypatch.setenv("PYTHON_EXECUTABLE", sys.executable)
  monkeypatch.delenv("SANDBOX_EXECUTION_UID", raising=False)
  monkeypatch.setattr(main.subprocess, "run", lambda *a, **kw: None)
  monkeypatch.setattr(main, "snapshot_work_tree", lambda *args: None)
  monkeypatch.setattr(main, "get_changed_files", lambda *args: {})
  monkeypatch.setattr(main, "_schedule_restart", lambda: None)
  yield main.app.test_client()
  common_tools._contaminated.clear()
  main._restart_scheduled.clear()


@pytest.mark.parametrize("route", ["/shell", "/run_code"])
@pytest.mark.parametrize("ending", ["success", "failure", "timeout"])
def test_both_routes_reset_on_every_exit(api, machine, route, ending):
  code = (
    "from pathlib import Path\nimport time, sys\n"
    f"Path({str(machine / 'etc/config')!r}).write_text('changed')\n"
    f"Path({str(machine / 'app/result')!r}).write_text('keep')\n"
  )
  code += {
    "success": "print('result')\n",
    "failure": "sys.exit(3)\n",
    "timeout": "time.sleep(60)\n",
  }[ending]
  timeout = 0.2 if ending == "timeout" else 5
  if route == "/run_code":
    payload = {"code": {"evaluator.py": code}, "timeout": timeout, "args": []}
  else:
    payload = {
      "cmd": f"{shlex.quote(sys.executable)} evaluator.py",
      "cwd": ".",
      "code_files": {"evaluator.py": code},
      "timeout": timeout,
    }
  response = api.post(route, json=payload)
  assert response.status_code in (200, 400), response.data
  assert (machine / "etc/config").read_text() == "setup"
  assert (machine / "app/result").read_text() == "keep"
  assert not main.reset_controller.active
  assert not main.reset_controller.reason
  assert "X-Sandbox-Retired" not in response.headers


@pytest.mark.parametrize("route", ["/shell", "/run_code"])
def test_completed_result_survives_reset_failure(api, controller, monkeypatch, route):
  original_finish = controller.finish

  def finish():
    result = original_finish()
    controller.retire("test cleanup incomplete")
    return result

  monkeypatch.setattr(controller, "finish", finish)
  code = 'print(\'{"status": "success", "result": {"fitness": 1}}\')'
  if route == "/run_code":
    payload = {"code": {"evaluator.py": code}, "timeout": 5, "args": []}
  else:
    payload = {
      "cmd": f"{shlex.quote(sys.executable)} evaluator.py",
      "cwd": ".",
      "code_files": {"evaluator.py": code},
      "timeout": 5,
    }

  response = api.post(route, json=payload)

  assert response.status_code == 200, response.data
  body = json.loads(response.get_data(as_text=True))
  if route == "/shell":
    body = json.loads(body["result"]["output"])
  assert body == {"status": "success", "result": {"fitness": 1}}
  assert response.headers["X-Sandbox-Retired"] == "1"


def test_incomplete_cleanup_blocks_next_command_and_health(api, controller, machine):
  controller.retire("test cleanup incomplete")
  response = api.post("/shell", json={"cmd": "unused", "cwd": ".", "code_files": {}})
  assert response.status_code == 503
  assert response.headers["X-Sandbox-Retired"] == "1"
  assert api.get("/health").status_code == 503


@pytest.mark.skipif(
  not sys.platform.startswith("linux")
  or os.environ.get("DAEMON_RESET_INTEGRATION") != "1",
  reason="native UID process discovery must run inside a disposable Linux container",
)
def test_native_detached_process_stopped_and_setup_process_preserved(machine):
  setup = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
  detached = None
  controller = SandboxReset(machine, machine / "tmp/.reset")
  try:
    controller.prepare()
    controller.begin()
    detached = subprocess.Popen(
      [sys.executable, "-c", "import time; time.sleep(60)"],
      start_new_session=True,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )
    assert controller.finish().status == "clean"
    detached.wait(timeout=5)
    assert detached.returncode < 0
    assert setup.poll() is None
  finally:
    for process in (setup, detached):
      if process is not None and process.poll() is None:
        process.kill()
      if process is not None:
        process.wait(timeout=5)
    controller.close()


@pytest.mark.skipif(
  not sys.platform.startswith("linux"), reason="inotify is Linux-only"
)
def test_large_untouched_image_and_repo_do_not_increase_reset_work(machine):
  image = machine / "usr/image"
  image.mkdir(parents=True)
  for index in range(5000):
    (image / str(index)).write_bytes(b"")
    (machine / "app" / str(index)).write_text("source")
  controller = SandboxReset(
    machine, machine / "tmp/.reset", excluded=(machine / "app",), track_processes=False
  )
  try:
    controller.prepare()
    assert controller.notifications.complete
    controller.begin()
    (machine / "etc/config").write_text("changed")
    result = controller.finish()
    assert result.status == "clean", result
    assert result.operations < 100
    assert (machine / "etc/config").read_text() == "setup"
    # Verify all baseline paths after losing notification coverage.
    controller.notifications.close()
    controller.notifications.complete = False
    controller.begin()
    (machine / "etc/config").write_text("changed again")
    result = controller.finish()
    assert result.status == "clean", result
    assert result.operations > 4096
    assert (machine / "etc/config").read_text() == "setup"
  finally:
    controller.close()

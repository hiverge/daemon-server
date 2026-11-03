"""Tests for the main module."""

import contextlib
import errno
import json
import logging
import os
import pathlib
import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest

import common_tools
import main
from conftest import wait_until_dead


@pytest.fixture
def linux_subreaper(monkeypatch):
  """Enable orphan adoption for one test and restore the runner's kernel setting."""
  if sys.platform != "linux":
    pytest.skip("Requires Linux child subreapers")
  # Match the production prctl declaration. Unsigned-long arguments also carry
  # the address used by PR_GET_CHILD_SUBREAPER without truncating it.
  libc = common_tools.ctypes.CDLL(None, use_errno=True)
  libc.prctl.argtypes = [common_tools.ctypes.c_int] + [common_tools.ctypes.c_ulong] * 4
  libc.prctl.restype = common_tools.ctypes.c_int
  original = common_tools.ctypes.c_int()
  # PR_GET_CHILD_SUBREAPER is 37 in Linux UAPI's <linux/prctl.h>.
  assert libc.prctl(37, common_tools.ctypes.addressof(original), 0, 0, 0) == 0
  monkeypatch.setattr(common_tools, "_subreaper_enabled", False)
  assert common_tools.enable_process_cleanup()
  # Assertions report cleanup failures without a restart timer exiting pytest.
  monkeypatch.setattr(main, "_restart_container", lambda: None)
  try:
    yield
  finally:
    assert libc.prctl(36, original.value, 0, 0, 0) == 0


@pytest.fixture(autouse=True)
def _scratch_directory(monkeypatch, tmp_path):
  (tmp_path / "scratch").mkdir()
  monkeypatch.setattr(main, "SCRATCH_DIR", str(tmp_path / "scratch"))
  monkeypatch.setattr(main, "TRACKING_GIT_DIR", str(tmp_path / ".agent_git"))
  monkeypatch.setattr(
    main, "REQUEST_TRACKING_GIT_DIR", str(tmp_path / ".agent_git.request")
  )
  monkeypatch.setattr(
    main, "SCRATCH_TRACKING_GIT_DIR", str(tmp_path / ".agent_git.scratch")
  )
  execute = main.execute_shell_command
  started = False

  def execute_after_setup(*args, **kwargs):
    nonlocal started
    # Tests populate their setup backup after fixtures run. Model daemon startup
    # immediately before the first shell execution, before applying any overlay.
    if not started:
      if not (pathlib.Path(main.TRACKING_GIT_DIR) / "index").exists():
        subprocess.run(
          ["rsync", "-a", "--delete", main.BACKUP_DIR, main.REPO_DIR], check=True
        )
        main.prepare_git_tracking()
      started = True
    return execute(*args, **kwargs)

  monkeypatch.setattr(main, "execute_shell_command", execute_after_setup)


class _Response:
  """
  Stand-in for the Flask response that `after_request` hooks receive and return.
  """


class _MockTimer:
  """
  Stand-in for `threading.Timer` that records the delay instead of arming a timer.
  """

  def __init__(self, delay: float, recorder: list[float]) -> None:
    self._delay = delay
    self._recorder = recorder

  def start(self) -> None:
    self._recorder.append(self._delay)


class TestEvaluationArguments:
  """
  Tests that the arguments a caller passes reach the evaluation script verbatim.

  `run_command` hands the command to `Popen` as a list with no shell, so the
  server must not quote or escape anything: whatever it puts in the list is
  exactly what lands in `sys.argv`.
  """

  # Echoes its own arguments back as the last line, i.e. the evaluator contract.
  _ECHO_ARGV = (
    "import json, sys\n"
    'print(json.dumps({"status": "success", "result": {"argv": sys.argv[1:]}}))\n'
  )

  @pytest.fixture(autouse=True)
  def _sandbox(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """
    Point the server at throwaway directories.
    """
    repo = tmp_path / "repo"
    backup = tmp_path / "backup"
    repo.mkdir()
    backup.mkdir()
    monkeypatch.setattr(main, "REPO_DIR", f"{repo}/")
    monkeypatch.setattr(main, "BACKUP_DIR", f"{backup}/")
    # The server resolves the interpreter from the environment at call time.
    monkeypatch.setenv("PYTHON_EXECUTABLE", sys.executable)
    # Stub the pre-run `rsync` restore: these tests are about what reaches
    # `sys.argv`, and rsync only exists in the sandbox image, not on every dev
    # machine. `run_command` uses `Popen`, so the evaluation itself still runs.
    run = subprocess.run

    def without_restore(args, *other, **kwargs):
      if args[0] == "rsync":
        return subprocess.CompletedProcess(args, 0)
      return run(args, *other, **kwargs)

    monkeypatch.setattr(main.subprocess, "run", without_restore)
    self.repo = repo

  def _argv_for(self, args: list) -> list[str]:
    """
    Run the echo script with `args` and return the `sys.argv[1:]` it saw.
    """
    output = main.execute_python_function(
      {"echo_argv.py": self._ECHO_ARGV}, args, 30, "echo_argv.py"
    )
    return json.loads(output)["result"]["argv"]

  def test_passes_arguments_through_unquoted(self) -> None:
    """
    Arguments arrive in `sys.argv` exactly as given, with no added quoting.
    """
    # given a mix of a bare value and a multi-token argument list.
    args = ["1", "subcommand", "-n", "512"]

    # when the evaluation script is run with them.
    argv = self._argv_for(args)

    # then it saw them verbatim.
    assert argv == ["1", "subcommand", "-n", "512"]
    assert not any('"' in arg for arg in argv)

  def test_stringifies_non_string_arguments(self) -> None:
    """
    A YAML-numeric argument reaches the script as its decimal string.
    """
    # given a numeric argument.
    args = [512]

    # when the evaluation script is run with it.
    argv = self._argv_for(args)

    # then it arrived as a plain string.
    assert argv == ["512"]

  def test_writes_code_files_before_running(self) -> None:
    """
    Files supplied with the request are in the repo by the time the script runs,
    so a caller can deliver an input file alongside the arguments naming it.
    """
    # given a script that reads a data file supplied in the same request.
    code_files = {
      "read_input.py": (
        "import json, sys\n"
        'print(json.dumps({"status": "success", '
        '"result": {"data": open(sys.argv[1]).read()}}))\n'
      ),
      "input.json": '["a", "b"]',
    }

    # when it is run with that file's path as its argument.
    output = main.execute_python_function(
      code_files, ["input.json"], 30, "read_input.py"
    )

    # then the script read the file the request delivered.
    assert json.loads(output)["result"]["data"] == '["a", "b"]'

  def test_deletes_tombstoned_files_before_running(self) -> None:
    """A null code-file value removes that path from the restored repo."""
    (self.repo / "obsolete.txt").write_text("baseline")
    script = (
      "import json, pathlib\n"
      'print(json.dumps({"status": "success", "result": '
      '{"deleted": not pathlib.Path("obsolete.txt").exists()}}))\n'
    )

    output = main.execute_python_function(
      {"check_deleted.py": script, "obsolete.txt": None},
      [],
      30,
      "check_deleted.py",
    )

    assert json.loads(output)["result"]["deleted"] is True

  def test_deleting_final_symlink_preserves_its_target(self) -> None:
    """A tombstone removes the named symlink rather than its target."""
    target = self.repo / "real.py"
    target.write_text("target")
    symlink = self.repo / "alias.py"
    symlink.symlink_to(target.name)

    main.execute_python_function(
      {"echo_argv.py": self._ECHO_ARGV, "alias.py": None},
      [],
      30,
      "echo_argv.py",
    )

    assert not symlink.is_symlink()
    assert target.read_text() == "target"

  @pytest.mark.parametrize("content", [None, "replacement"])
  def test_rejects_code_file_paths_outside_repo(self, content: str | None) -> None:
    """Overlay writes and deletions cannot escape through parent traversal."""
    outside = self.repo.parent / "outside.txt"
    outside.write_text("keep me")

    with pytest.raises(ValueError, match="Path escapes repo directory"):
      main.execute_python_function({"../outside.txt": content}, [], 30, "unused.py")

    assert outside.read_text() == "keep me"

  def test_rejects_code_file_paths_through_external_symlink(
    self,
  ) -> None:
    """An in-repo symlinked parent cannot redirect a deletion outside."""
    outside_dir = self.repo.parent / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "outside.txt"
    outside.write_text("keep me")
    (self.repo / "escape").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="Path escapes repo directory"):
      main.execute_python_function({"escape/outside.txt": None}, [], 30, "unused.py")

    assert outside.read_text() == "keep me"

  @pytest.mark.parametrize(
    ("route", "payload"),
    [
      (
        "/run_code",
        {"code": {"../outside.txt": None}, "timeout": 30},
      ),
      (
        "/shell",
        {
          "cmd": "true",
          "cwd": ".",
          "code_files": {"../outside.txt": None},
        },
      ),
    ],
  )
  def test_path_escape_returns_structured_client_error(
    self, route: str, payload: dict
  ) -> None:
    """Both endpoints return validation failures as JSON 400 responses."""
    response = main.app.test_client().post(route, json=payload)

    assert response.status_code == 400
    assert response.get_json() == {
      "status": "failed",
      "error": "Path escapes repo directory: ../outside.txt",
    }

  def test_creates_missing_directories_for_code_files(self) -> None:
    """
    A code file in a directory the repo does not have yet is written, as
    /shell already does, rather than failing the request.
    """
    # given a script that reads a file delivered into a new directory.
    code_files = {
      "read_input.py": (
        "import json\n"
        'print(json.dumps({"status": "success", '
        '"result": {"data": open("new/dir/input.txt").read()}}))\n'
      ),
      "new/dir/input.txt": "delivered",
    }

    # when it is run.
    output = main.execute_python_function(code_files, [], 30, "read_input.py")

    # then the file was there.
    assert json.loads(output)["result"]["data"] == "delivered"

  @pytest.mark.parametrize(
    ("route", "payload"),
    [
      (
        "/run_code",
        {"code": {"a_file/inside.txt": "x"}, "timeout": 30},
      ),
      (
        "/shell",
        {"cmd": "true", "cwd": ".", "code_files": {"a_file/inside.txt": "x"}},
      ),
    ],
  )
  def test_unwritable_code_file_returns_structured_client_error(
    self, route: str, payload: dict
  ) -> None:
    """
    A code file that cannot be written (here: under a path that is a file)
    is a JSON 400 naming the file, not an HTML 500.
    """
    # given a repo where `a_file` is a file, not a directory.
    (self.repo / "a_file").write_text("plain file")

    # when a request delivers a file inside it.
    response = main.app.test_client().post(route, json=payload)

    # then the caller is told which file could not be written.
    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "failed"
    assert body["error"].startswith("Cannot write code file a_file/inside.txt:")


class TestShellCommandDescendants:
  """
  Tests that a shell command backgrounding work neither holds the request open
  for the lifetime of that work nor leaves it running on the pod.

  A backgrounded job inherits the command's stdout/stderr, so the pipes end only
  once it too has exited.
  """

  @pytest.fixture(autouse=True)
  def _sandbox(
    self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
  ) -> pathlib.Path:
    """
    Point the server at throwaway directories and shorten the reader deadline.
    """
    repo = tmp_path / "repo"
    backup = tmp_path / "backup"
    repo.mkdir()
    backup.mkdir()
    monkeypatch.setattr(main, "REPO_DIR", f"{repo}/")
    monkeypatch.setattr(main, "BACKUP_DIR", f"{backup}/")
    monkeypatch.setattr(main, "TRACKING_GIT_DIR", str(tmp_path / ".agent_git"))
    monkeypatch.setattr(common_tools, "READER_JOIN_TIMEOUT", 1.0)
    # Process-wide, so it outlives a single test.
    common_tools._contaminated.clear()
    self.repo = repo
    self.backup = backup
    return repo

  def _run(self, cmd: str, timeout: float = 30) -> str:
    """
    Run `cmd` in the sandbox repo and return only its output.
    """
    output, _ = main.execute_shell_command(cmd, str(self.repo), {}, timeout)
    return output

  def test_deletes_tombstoned_files_before_running(self) -> None:
    """A null code-file value removes that path from the restored repo."""
    (self.backup / "obsolete.txt").write_text("baseline")

    output, _ = main.execute_shell_command(
      "test ! -e obsolete.txt && echo deleted",
      str(self.repo),
      {"obsolete.txt": None},
      30,
    )

    assert output.strip() == "deleted"

  def test_deleting_final_symlink_preserves_its_target(self) -> None:
    """Shell overlays also unlink the named symlink, not its target."""
    target = self.backup / "real.py"
    target.write_text("target")
    (self.backup / "alias.py").symlink_to(target.name)

    main.execute_shell_command(
      "true",
      str(self.repo),
      {"alias.py": None},
      30,
    )

    assert not (self.repo / "alias.py").is_symlink()
    assert (self.repo / "real.py").read_text() == "target"

  def test_rejects_tombstoned_file_outside_repo(self) -> None:
    """Shell overlays use the same repository containment check."""
    outside = self.repo.parent / "outside.txt"
    outside.write_text("keep me")

    with pytest.raises(ValueError, match="Path escapes repo directory"):
      main.execute_shell_command("true", str(self.repo), {"../outside.txt": None}, 30)

    assert outside.read_text() == "keep me"

  def test_returns_when_the_command_exits_not_when_its_background_job_does(
    self,
  ) -> None:
    """
    A command that backgrounds a long-running job returns as soon as the command
    itself exits.
    """
    # given a command that backgrounds a 30 second job and then exits.
    cmd = "sleep 30 & echo started"

    # when it is run.
    start = time.monotonic()
    output = self._run(cmd)
    elapsed = time.monotonic() - start

    # then it returns on the command's own exit, not the job's 30s lifetime.
    assert elapsed < 10
    assert "started" in output

  def test_kills_the_job_the_command_backgrounded(self) -> None:
    """
    Work a command leaves running in the background is killed with it, so it
    cannot compete with the next request for the pod.
    """
    # given a command that backgrounds a 30 second job and reports its pid.
    cmd = "sleep 30 & echo $!"

    # when it is run.
    output = self._run(cmd)

    # then the backgrounded job was killed.
    job_pid = int(output.strip())
    assert wait_until_dead(job_pid), (
      f"backgrounded job {job_pid} was left running after the request"
    )

  def test_kills_the_background_job_of_a_timed_out_command(
    self, tmp_path: pathlib.Path
  ) -> None:
    """
    Timing out kills the whole process group, so a command that hangs leaves
    nothing behind.
    """
    # given a command that backgrounds a 30 second job and then hangs itself. The
    # pid goes to a file, as a timed-out command reports only the timeout.
    pid_file = tmp_path / "job.pid"
    cmd = f"sleep 30 & echo $! > {pid_file}; sleep 30"

    # when it is run with a short timeout.
    output = self._run(cmd, timeout=1)

    # then the timeout is reported and the backgrounded job was killed with it.
    assert output == "Error: command timed out after 1s"
    job_pid = int(pid_file.read_text())
    assert wait_until_dead(job_pid), f"backgrounded job {job_pid} survived the timeout"

  def test_reports_output_as_truncated_when_it_cannot_be_drained(
    self, tmp_path: pathlib.Path
  ) -> None:
    """
    Output still held by a surviving descendant is reported as truncated, and the
    output that did arrive is still returned.
    """
    # given a command whose child leaves the process group, so it survives the
    # kill and keeps holding the inherited pipes for longer than any deadline in
    # play. The pid goes to a file so this test can clean up what the code under
    # test cannot.
    pid_file = tmp_path / "escaper.pid"
    cmd = (
      f"{sys.executable} -c "
      "'import subprocess, sys, os; "
      "child = subprocess.Popen("
      '  [sys.executable, "-c", "import time; time.sleep(30)"], '
      "  preexec_fn=os.setsid); "
      f'open("{pid_file}", "w").write(str(child.pid)); '
      'print("partial output")\''
    )

    # when it is run.
    start = time.monotonic()
    try:
      output = self._run(cmd)
      elapsed = time.monotonic() - start
    finally:
      if pid_file.exists():
        with contextlib.suppress(ProcessLookupError):
          os.kill(int(pid_file.read_text()), signal.SIGKILL)

    # then the output that arrived is returned, flagged as incomplete.
    assert "partial output" in output
    assert "[Output truncated" in output

    # and it returned on its own deadline, nowhere near the escapee's 30s.
    assert elapsed < 10, f"took {elapsed:.1f}s, so cleanup waited on the escapee"

  def test_does_not_log_the_command_output(
    self, caplog: pytest.LogCaptureFixture
  ) -> None:
    """
    A shell command's output is kept out of the sandbox logs, so an agent that
    reads a source file through the tool cannot leak its contents there.
    """
    # given a command that prints a line standing in for file contents.
    cmd = "echo 'secret source line'"

    # when it is run.
    with caplog.at_level(logging.INFO):
      output = self._run(cmd)

    # then the output is returned to the caller but never logged.
    assert output.strip() == "secret source line"
    assert not any(
      "secret source line" in record.getMessage() for record in caplog.records
    )

  def test_schedules_a_restart_once_the_pod_is_contaminated(
    self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    """
    A survivor triggers a container restart.
    """
    # given a contaminated pod, and a stand-in for the timer so the exit never
    # runs here.
    restarts: list[float] = []
    monkeypatch.setattr(
      main.threading, "Timer", lambda delay, fn: _MockTimer(delay, restarts)
    )
    common_tools.mark_contaminated()

    # when a response goes out.
    main._restart_if_contaminated(_Response())

    # then a restart was scheduled at the grace delay.
    assert restarts == [main._RESTART_GRACE]


class TestShellChangeTracking:
  """Report each command's changes using a fresh copy of the startup index."""

  @pytest.fixture(autouse=True)
  def _sandbox(
    self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
  ) -> pathlib.Path:
    """
    Point the server at throwaway directories and shorten the reader deadline.
    """
    repo = tmp_path / "repo"
    backup = tmp_path / "backup"
    repo.mkdir()
    backup.mkdir()
    monkeypatch.setattr(main, "REPO_DIR", f"{repo}/")
    monkeypatch.setattr(main, "BACKUP_DIR", f"{backup}/")
    monkeypatch.setattr(main, "TRACKING_GIT_DIR", str(tmp_path / ".agent_git"))
    monkeypatch.setattr(common_tools, "READER_JOIN_TIMEOUT", 1.0)
    common_tools._contaminated.clear()
    self.repo = repo
    self.backup = backup
    self.tracking = tmp_path / ".agent_git"
    return repo

  def _changed(self, cmd: str) -> dict:
    """
    Run `cmd` in the sandbox repo and return only what it changed.
    """
    _, files = main.execute_shell_command(cmd, str(self.repo), {}, 30)
    return files

  def test_modified_files_are_reported_with_their_new_content(self) -> None:
    """An edit to a baseline file comes back carrying the post-command text."""
    # given a baseline file, and a second one the command leaves alone.
    (self.backup / "edit.txt").write_text("before")
    (self.backup / "kept.txt").write_text("untouched")

    # when the command overwrites only the first.
    files = self._changed("echo after > edit.txt")

    # then the edit is reported and the untouched file is not.
    assert files == {"edit.txt": "after\n"}

  def test_created_files_are_reported(self) -> None:
    """A file the command creates is reported, nested paths included."""
    # given an empty baseline.
    # when the command writes a new file at the root and one in a new directory.
    files = self._changed(
      "echo fresh > new.txt; mkdir -p out && echo built > out/artifact.log"
    )

    # then both are reported under their repo-relative paths.
    assert files == {"new.txt": "fresh\n", "out/artifact.log": "built\n"}

  def test_deleted_files_are_reported_as_none(self) -> None:
    """A deletion is signalled by a None content rather than by omission."""
    # given a baseline file the command will remove.
    (self.backup / "remove.txt").write_text("doomed")

    # when the command deletes it.
    files = self._changed("rm remove.txt")

    # then it is reported with no content.
    assert files == {"remove.txt": None}

  def test_binary_files_are_not_reported(self) -> None:
    """
    Undecodable content has no text to return, so it is omitted rather than
    failing the call.
    """
    # given an empty baseline.
    write_blob = (
      f'{sys.executable} -c \'open("blob.bin", "wb").write(b"\\xff\\xfe\\x00binary")\''
    )
    files = self._changed(f"{write_blob}; echo real > kept.txt")

    # then the text file comes back and the binary is left out.
    assert files == {"kept.txt": "real\n"}

  def test_changes_do_not_persist_into_the_next_call(self) -> None:
    """Each request stages its restored tree against the fixed setup template."""
    # given a first call that edits one baseline file and deletes another.
    (self.backup / "edit.txt").write_text("before")
    (self.backup / "remove.txt").write_text("doomed")
    assert self._changed("echo after > edit.txt; rm remove.txt")

    # when a second call changes nothing, and a third changes one file.
    quiet = self._changed("true")
    single = self._changed("echo third > edit.txt")

    # then the previous call's work has been forgotten, and the new edit is seen
    # against the restored content rather than against the stale index entry.
    assert quiet == {}
    assert single == {"edit.txt": "third\n"}

  def test_request_overlays_are_the_comparison_baseline(self):
    """Different overlays on one worker each define their own snapshot A."""
    (self.backup / "edit.txt").write_text("setup")

    _, quiet = main.execute_shell_command(
      "true", str(self.repo), {"edit.txt": "first"}, 30
    )
    _, changed = main.execute_shell_command(
      "printf after > edit.txt", str(self.repo), {"edit.txt": "first"}, 30
    )
    _, next_quiet = main.execute_shell_command(
      "true", str(self.repo), {"edit.txt": "second"}, 30
    )

    assert quiet == next_quiet == {}
    assert changed == {"edit.txt": "after"}

  def test_baseline_metadata_and_objects_do_not_change_between_requests(self):
    """Only the request store receives overlay blobs and index updates."""
    (self.backup / "edit.txt").write_text("setup")
    assert self._changed("true") == {}

    def baseline_files():
      return {
        path.relative_to(self.tracking): path.read_bytes()
        for path in self.tracking.rglob("*")
        if path.is_file()
      }

    prepared = baseline_files()
    main.execute_shell_command(
      "printf result > edit.txt", str(self.repo), {"edit.txt": "overlay"}, 30
    )
    assert self._changed("true") == {}

    assert baseline_files() == prepared
    assert not pathlib.Path(main.REQUEST_TRACKING_GIT_DIR).exists()

  def test_request_reads_setup_blobs_from_the_baseline_object_store(self):
    """A clean request resolves setup blobs without its own copies."""
    (self.backup / "kept.txt").write_text("setup")
    self._changed("true")
    staged = main._run_tracking_git(
      ["ls-files", "--stage", "--", "kept.txt"],
      main.REPO_DIR,
      main.TRACKING_GIT_DIR,
    ).stdout
    blob = staged.split()[1]
    request_git = shlex.quote(main.REQUEST_TRACKING_GIT_DIR)
    output, files = main.execute_shell_command(
      f"git --git-dir {request_git} cat-file blob {blob}; "
      f"git --git-dir {request_git} count-objects",
      str(self.repo),
      {},
      30,
    )

    assert output.startswith("setup0 objects")
    assert files == {}
    assert not pathlib.Path(main.REQUEST_TRACKING_GIT_DIR).exists()

  def test_tracking_excludes_and_index_flags_do_not_leak_to_the_next_request(self):
    """Changes to a request's own Git policy are discarded with its metadata."""
    (self.backup / "edit.txt").write_text("setup")
    tracking = shlex.quote(main.REQUEST_TRACKING_GIT_DIR)
    self._changed(
      f"mkdir -p {tracking}/info; "
      f"printf 'new.txt\\n' > {tracking}/info/exclude; "
      f"git --git-dir {tracking} --work-tree {shlex.quote(str(self.repo))} "
      "update-index --skip-worktree edit.txt"
    )

    changed = self._changed("printf changed > edit.txt; printf fresh > new.txt")

    assert changed == {"edit.txt": "changed", "new.txt": "fresh"}

  def test_global_git_configuration_does_not_control_tracking(
    self, monkeypatch, tmp_path
  ):
    """Tracking policy is independent of configuration outside the workspace."""
    excludes = tmp_path / "excludes"
    excludes.write_text("new.txt\n")
    config = tmp_path / "global-git-config"
    config.write_text(f"[core]\n\texcludesFile = {excludes}\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))

    assert self._changed("printf fresh > new.txt") == {"new.txt": "fresh"}

  @pytest.mark.parametrize(
    ("cmd", "timeout"),
    [("true", 30), ("exit 7", 30), ("sleep 30", 0.1)],
  )
  def test_http_request_discards_both_tracking_repositories(self, cmd, timeout):
    """Success, command failure, and timeout all end the request's Git lifetime."""
    response = main.app.test_client().post(
      "/shell",
      json={"cwd": ".", "cmd": cmd, "code_files": {}, "timeout": timeout},
    )

    assert response.status_code == 200
    assert (self.tracking / "index").is_file()
    assert not pathlib.Path(main.REQUEST_TRACKING_GIT_DIR).exists()
    assert not pathlib.Path(main.SCRATCH_TRACKING_GIT_DIR).exists()

  def test_tracking_failure_discards_request_state_and_next_call_recovers(self):
    """A failed comparison cannot leave corrupt metadata for the next call."""
    tracking = shlex.quote(main.REQUEST_TRACKING_GIT_DIR)
    response = main.app.test_client().post(
      "/shell",
      json={
        "cwd": ".",
        "cmd": f"printf '[' > {tracking}/config",
        "code_files": {},
      },
    )

    assert response.status_code == 500
    assert not pathlib.Path(main.REQUEST_TRACKING_GIT_DIR).exists()
    assert not pathlib.Path(main.SCRATCH_TRACKING_GIT_DIR).exists()
    assert self._changed("printf recovered > new.txt") == {"new.txt": "recovered"}

  def test_scratch_round_trip_and_worker_reuse(self):
    client = main.app.test_client()

    def run(cwd, cmd, **fields):
      response = client.post(
        "/shell",
        json={"cwd": cwd, "cmd": cmd, "code_files": {}, **fields},
      )
      assert response.status_code == 200
      return response.get_json()["result"]

    first = run(
      "../scratch",
      "cat notes.md; printf after > notes.md; "
      "printf new > new.txt; printf candidate > ../repo/main.py",
      scratch_files={"notes.md": "before", "kept.md": "unchanged"},
    )
    assert first == {
      "output": "before",
      "files": {"main.py": "candidate"},
      "scratch_files": {"notes.md": "after", "new.txt": "new"},
    }
    snapshot = {"kept.md": "unchanged", **first["scratch_files"]}
    second = run(main.SCRATCH_DIR, "cat notes.md", scratch_files=snapshot)
    assert second == {"output": "after", "files": {}, "scratch_files": {}}
    fresh = run(
      f"{main.SCRATCH_DIR}/../repo",
      "test ! -e ../scratch/notes.md && echo fresh",
    )
    assert fresh == {"output": "fresh\n", "files": {}, "scratch_files": {}}
    assert list(pathlib.Path(main.SCRATCH_DIR).iterdir()) == []
    assert not pathlib.Path(main.SCRATCH_TRACKING_GIT_DIR).exists()

  @pytest.mark.parametrize(
    "fields", [{"cwd": ".."}, {"scratch_files": {"../outside.txt": "bad"}}]
  )
  def test_scratch_request_boundaries(self, fields):
    response = main.app.test_client().post(
      "/shell", json={"cwd": ".", "cmd": "true", "code_files": {}, **fields}
    )
    assert response.status_code == 400
    assert response.get_json()["status"] == "failed"
    assert not (self.repo.parent / "outside.txt").exists()
    assert list(pathlib.Path(main.SCRATCH_DIR).iterdir()) == []


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
class TestTemporaryEntryCleanup:
  """Exercise cleanup in private roots, including both HTTP execution paths."""

  @pytest.fixture(autouse=True)
  def _sandbox(self, monkeypatch, tmp_path):
    self.root = tmp_path / "filesystem_root"
    self.root.mkdir()
    self.roots = tuple(self.root / name for name in ("tmp", "var/tmp", "dev/shm"))
    for root in self.roots:
      root.mkdir(parents=True)
    self.repo = self.root / "app"
    self.backup = self.root / "shared" / "repo"
    self.scratch = self.root / "scratch"
    self.repo.mkdir()
    self.backup.mkdir(parents=True)
    self.scratch.mkdir()
    self.tracking = self.roots[0] / ".agent_git"
    monkeypatch.setattr(main, "_ROOT_DIR", self.root)
    monkeypatch.setattr(main, "_TEMPORARY_DIRS", self.roots)
    monkeypatch.setattr(main, "_temporary_baseline", {})
    monkeypatch.setattr(main, "_temporary_events", None)
    monkeypatch.setattr(main, "_temporary_excluded", frozenset())
    monkeypatch.setattr(main, "_temporary_virtual", frozenset())
    monkeypatch.setattr(main, "_filesystem_contamination_reason", None)
    monkeypatch.setattr(main, "REPO_DIR", f"{self.repo}/")
    monkeypatch.setattr(main, "BACKUP_DIR", f"{self.backup}/")
    monkeypatch.setattr(main, "SCRATCH_DIR", str(self.scratch))
    monkeypatch.setattr(main, "TRACKING_GIT_DIR", str(self.tracking))
    monkeypatch.setattr(main, "REQUEST_TRACKING_GIT_DIR", f"{self.tracking}.request")
    monkeypatch.setattr(main, "SCRATCH_TRACKING_GIT_DIR", f"{self.tracking}.scratch")
    monkeypatch.setenv("PYTHON_EXECUTABLE", sys.executable)
    common_tools._contaminated.clear()
    yield
    if main._temporary_events is not None:
      main._temporary_events.close()

  def _capture(self):
    main._temporary_baseline = main._capture_temporary_baseline()

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  @pytest.mark.parametrize(
    "mutation", ["write", "truncate", "delete", "chmod", "replace", "rename"]
  )
  def test_records_setup_file_changes_without_scanning(self, monkeypatch, mutation):
    installed = self.root / "opt" / "runtime" / "lib"
    installed.mkdir(parents=True)
    original = installed / "library.so"
    original.write_text("setup")
    self._capture()
    assert main._filesystem_contamination_reason is None
    if mutation == "write":
      original.write_text("request")
    elif mutation == "truncate":
      with original.open("w"):
        pass
    elif mutation == "delete":
      original.unlink()
    elif mutation == "chmod":
      original.chmod(0o600)
    elif mutation == "replace":
      replacement = self.repo / "library.so"
      replacement.write_text("request")
      replacement.replace(original)
    else:
      original.rename(self.repo / "library.so")

    def no_scan(*args, **kwargs):
      pytest.fail("An existing-file mutation required a directory scan.")

    # Replacements create an addition event and use normal parent reconciliation.
    if mutation != "replace":
      monkeypatch.setattr(main.os, "scandir", no_scan)
    main._cleanup_temporary_entries()

    assert str(original) in main._filesystem_contamination_reason
    assert not common_tools.is_contaminated()
    if mutation in ("write", "replace"):
      assert original.read_text() == "request"

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  @pytest.mark.parametrize("mutation", ["chmod", "delete", "replace_symlink"])
  def test_records_setup_directory_and_symlink_changes(self, mutation):
    original = self.root / "opt" / "runtime"
    original.mkdir(parents=True)
    link = original / "library-link"
    link.symlink_to(self.repo / "library.so")
    self._capture()
    if mutation == "chmod":
      original.chmod(0o700)
    elif mutation == "delete":
      shutil.rmtree(original)
    else:
      link.unlink()
      link.symlink_to(self.repo / "other.so")

    main._cleanup_temporary_entries()

    assert main._filesystem_contamination_reason is not None
    if mutation == "replace_symlink":
      assert str(link) in main._filesystem_contamination_reason
      assert link.is_symlink()

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_new_artifacts_and_workspace_churn_do_not_flag_contamination(self):
    installed = self.root / "opt" / "runtime"
    installed.mkdir(parents=True)
    baseline_file = installed / "library.so"
    baseline_file.write_text("setup")
    self._capture()
    artifact = installed / "new.txt"
    artifact.write_text("first")
    artifact.write_text("second")
    artifact.chmod(0o600)
    artifact.rename(installed / "renamed.txt")
    new_tree = installed / "new-package" / "nested"
    new_tree.mkdir(parents=True)
    (new_tree / "result").write_text("request")
    removed = installed / "already-removed.txt"
    removed.write_text("request")
    removed.unlink()
    for workspace in (
      self.repo,
      self.backup,
      self.scratch,
      self.tracking,
      pathlib.Path(main.REQUEST_TRACKING_GIT_DIR),
      pathlib.Path(main.SCRATCH_TRACKING_GIT_DIR),
    ):
      workspace.mkdir(exist_ok=True)
      (workspace / "state").write_text("first")
      (workspace / "state").write_text("second")
    shutil.rmtree(self.scratch)
    self.scratch.mkdir()

    main._cleanup_temporary_entries()
    # The next drain includes deletion events generated by cleanup itself.
    main._cleanup_temporary_entries()

    assert {entry.name for entry in installed.iterdir()} == {"library.so"}
    assert baseline_file.read_text() == "setup"
    assert main._filesystem_contamination_reason is None

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_contamination_reason_survives_later_quiet_requests(self, caplog):
    original = self.roots[0] / "setup.txt"
    original.write_text("setup")
    self._capture()
    with caplog.at_level(logging.WARNING):
      original.write_text("request")
      main._cleanup_temporary_entries()
      reason = main._filesystem_contamination_reason
      main._cleanup_temporary_entries()
      original.chmod(0o600)
      main._cleanup_temporary_entries()

    assert reason is not None
    assert main._filesystem_contamination_reason == reason
    assert (
      sum(
        record.message.startswith("Filesystem contamination:")
        for record in caplog.records
      )
      == 1
    )

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  @pytest.mark.parametrize("route", ["/shell", "/run_code"])
  def test_http_execution_records_mutation_and_preserves_response(
    self, route, monkeypatch
  ):
    original = self.root / "opt" / "library.so"
    original.parent.mkdir()
    original.write_text("setup")
    self._capture()
    script = (
      "from pathlib import Path\n"
      f"Path({str(original)!r}).write_text('request')\n"
      "print('completed')\n"
    )
    payload = (
      {
        "cmd": f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
        "cwd": ".",
        "code_files": {},
      }
      if route == "/shell"
      else {
        "code": {"probe.py": script},
        "evaluation_script": "probe.py",
        "timeout": 30,
      }
    )

    def no_restart(*args, **kwargs):
      pytest.fail("Filesystem detection scheduled a container restart.")

    monkeypatch.setattr(main.threading, "Timer", no_restart)
    response = main.app.test_client().post(route, json=payload)

    assert response.status_code == 200
    assert "completed" in response.get_data(as_text=True)
    assert str(original) in main._filesystem_contamination_reason
    assert original.read_text() == "request"
    assert not common_tools.is_contaminated()
    assert not main.sandbox_lock.locked()

  def test_removes_new_entries_and_preserves_workspace_and_virtual_trees(
    self, monkeypatch
  ):
    image_trees = tuple(self.root / name for name in ("app", "opt", "proc", "sys"))
    for tree in image_trees:
      (tree / "nested").mkdir(parents=True)
      (tree / "nested" / "setup.txt").write_text("setup")
    kept = self.root / "setup.txt"
    kept.write_text("setup")
    protected = (self.repo, self.backup, self.scratch, *image_trees[2:])
    identities = {(path.stat().st_dev, path.stat().st_ino) for path in protected}
    real_scandir = os.scandir

    def avoid_protected_trees(path):
      info = os.fstat(path) if isinstance(path, int) else os.stat(path)
      assert (info.st_dev, info.st_ino) not in identities
      return real_scandir(path)

    monkeypatch.setattr(main.os, "scandir", avoid_protected_trees)
    self._capture()
    root_names = {entry.name for entry in self.root.iterdir()}
    kept.write_text("modified")
    for tree in image_trees:
      (tree / "nested" / "new.txt").write_text("preserved")
    (self.root / "new.log").write_text("remove")
    new_tree = self.root / "new-folder"
    (new_tree / "nested").mkdir(parents=True)
    (new_tree / "nested" / "output").write_text("remove")
    (self.root / "new_link").symlink_to(image_trees[1], target_is_directory=True)
    (self.root / "broken_link").symlink_to(self.root / "missing")
    nested_temporary = self.roots[0] / "cache"
    nested_temporary.mkdir()

    main._cleanup_temporary_entries()

    assert {entry.name for entry in self.root.iterdir()} == root_names
    assert kept.read_text() == "modified"
    for tree in image_trees:
      assert (tree / "nested" / "setup.txt").read_text() == "setup"
      if tree == self.root / "opt":
        assert not (tree / "nested" / "new.txt").exists()
      else:
        assert (tree / "nested" / "new.txt").read_text() == "preserved"
    assert not nested_temporary.exists()

  @pytest.mark.parametrize(
    "configured_path", ["REPO_DIR", "BACKUP_DIR", "SCRATCH_DIR", "TRACKING_GIT_DIR"]
  )
  def test_preserves_configured_root_entries_created_after_capture(
    self, monkeypatch, configured_path
  ):
    protected = self.root / "protected"
    monkeypatch.setattr(main, configured_path, str(protected))
    self._capture()
    protected.mkdir()
    (protected / "state").write_text("preserved")
    (self.root / "probe.log").write_text("remove")

    main._cleanup_temporary_entries()

    assert (protected / "state").read_text() == "preserved"
    assert not (self.root / "probe.log").exists()

  @pytest.mark.parametrize(
    "directory",
    [
      "opt/runtime/lib",
      "usr/local/lib/package",
      "etc/service",
      "home/user/.cache",
      "shared/runtime",
    ],
  )
  def test_cleans_deep_image_additions_and_retains_existing_contents(self, directory):
    parent = self.root / directory
    parent.mkdir(parents=True)
    baseline_file = parent / "setup.txt"
    baseline_file.write_text("setup")
    self._capture()
    baseline_file.write_text("modified")
    (parent / "new.txt").write_text("remove")
    new_tree = parent / "new-package" / "nested"
    new_tree.mkdir(parents=True)
    (new_tree / "output").write_text("remove")
    (parent / "workspace-link").symlink_to(self.repo, target_is_directory=True)
    (self.repo / "keep.txt").write_text("workspace")

    main._cleanup_temporary_entries()

    assert {entry.name for entry in parent.iterdir()} == {"setup.txt"}
    assert baseline_file.read_text() == "modified"
    assert (self.repo / "keep.txt").read_text() == "workspace"
    assert not common_tools.is_contaminated()

  def test_skips_device_tree_but_cleans_shared_memory(self, monkeypatch):
    devices = self.root / "dev"
    setup = self.roots[2] / "setup"
    setup.mkdir()
    (setup / "keep.txt").write_text("setup")
    device_identity = (devices.stat().st_dev, devices.stat().st_ino)
    real_scandir = os.scandir

    def avoid_devices(path):
      info = os.fstat(path) if isinstance(path, int) else os.stat(path)
      assert (info.st_dev, info.st_ino) != device_identity
      return real_scandir(path)

    monkeypatch.setattr(main.os, "scandir", avoid_devices)
    self._capture()
    (devices / "runtime-device").write_text("preserved")
    (setup / "new.txt").write_text("remove")

    main._cleanup_temporary_entries()

    assert (devices / "runtime-device").read_text() == "preserved"
    assert (setup / "keep.txt").read_text() == "setup"
    assert not (setup / "new.txt").exists()

  @pytest.mark.parametrize("root_index", [0, 1, 2])
  def test_removes_additions_and_preserves_setup_files(self, root_index):
    root = self.roots[root_index]
    kept = root / "setup.txt"
    kept.write_text("setup")
    setup = root / "setup_tree"
    setup.mkdir()
    (setup / "source.py").write_text("setup")
    outside = self.repo / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (root / "setup_link").symlink_to(outside, target_is_directory=True)
    self._capture()

    # Setup file contents are retained; new names inside setup trees are removed.
    kept.write_text("modified")
    (setup / "new.txt").write_text("inside setup tree")
    (root / "new.log").write_text("new")
    new_tree = root / "new_tree"
    (new_tree / "nested").mkdir(parents=True)
    (new_tree / "nested" / "output").write_text("new")
    (new_tree / "alias").symlink_to(outside, target_is_directory=True)
    (root / "new_link").symlink_to(outside, target_is_directory=True)
    (root / "broken_link").symlink_to(root / "missing")

    main._cleanup_temporary_entries()

    assert {entry.name for entry in root.iterdir()} == {
      "setup.txt",
      "setup_tree",
      "setup_link",
    }
    assert kept.read_text() == "modified"
    assert (setup / "source.py").read_text() == "setup"
    assert not (setup / "new.txt").exists()
    assert (root / "setup_link").is_symlink()
    assert (outside / "keep.txt").read_text() == "keep"

    (root / "next.log").write_text("next request")
    main._cleanup_temporary_entries()
    assert not (root / "next.log").exists()

  def test_preserves_tracking_created_after_capture(self):
    self._capture()
    self.tracking.mkdir()
    (self.tracking / "index").write_text("cached index")
    (self.roots[0] / "probe.log").write_text("new")

    main._cleanup_temporary_entries()

    assert (self.tracking / "index").read_text() == "cached index"
    assert not (self.roots[0] / "probe.log").exists()

  def test_accepts_root_aliases_and_skips_absent_roots(self, monkeypatch):
    alias = self.repo / "tmp_alias"
    alias.symlink_to(self.roots[0], target_is_directory=True)
    absent = self.repo / "absent"
    monkeypatch.setattr(main, "_TEMPORARY_DIRS", (alias, absent))
    self._capture()
    (self.roots[0] / "new.log").write_text("new")

    main._cleanup_temporary_entries()

    assert not (self.roots[0] / "new.log").exists()
    assert not absent.exists()

  def test_logs_removal_failure_without_queuing_retries(self, monkeypatch, caplog):
    self._capture()
    blocked = self.roots[0] / "blocked"
    blocked.write_text("keep")
    removable = self.roots[0] / "removable"
    removable.write_text("remove")
    real_unlink = os.unlink

    def deny_blocked(path, **kwargs):
      if path == blocked.name:
        raise PermissionError("removal denied")
      return real_unlink(path, **kwargs)

    monkeypatch.setattr(main.os, "unlink", deny_blocked)
    with caplog.at_level(logging.WARNING):
      main._cleanup_temporary_entries()

    assert blocked.exists()
    assert not removable.exists()
    assert "Could not remove new filesystem entry" in caplog.text
    assert str(blocked) in main._filesystem_contamination_reason
    reason = main._filesystem_contamination_reason
    monkeypatch.setattr(main.os, "unlink", real_unlink)
    main._cleanup_temporary_entries()
    assert blocked.exists()
    assert main._filesystem_contamination_reason == reason

  @pytest.mark.parametrize(
    "route, exit_code, lose_coverage",
    [
      ("/shell", 0, False),
      ("/shell", 7, False),
      ("/run_code", 0, False),
      ("/run_code", 7, False),
      ("/shell", 0, True),
    ],
  )
  def test_preserves_response_after_cleanup_or_coverage_loss(
    self, route, exit_code, lose_coverage
  ):
    nested = self.roots[0] / "setup_cache" / "nested"
    nested.mkdir(parents=True)
    self._capture()
    artifact = self.roots[0] / "probe.log"
    nested_artifact = nested / "probe.log"
    root_artifact = self.root / "probe.log"
    move_directory = (
      f"Path({str(nested)!r}).rename({str(self.repo / 'moved')!r})\n"
      if lose_coverage
      else ""
    )
    script = (
      "from pathlib import Path\n"
      f"Path({str(artifact)!r}).write_text('created')\n"
      f"Path({str(nested_artifact)!r}).write_text('created')\n"
      f"Path({str(root_artifact)!r}).write_text('created')\n"
      f"{move_directory}"
      "print('completed')\n"
      f"raise SystemExit({exit_code})\n"
    )
    if route == "/shell":
      payload = {
        "cmd": f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
        "cwd": ".",
        "code_files": {},
      }
    else:
      payload = {
        "code": {"probe.py": script},
        "evaluation_script": "probe.py",
        "timeout": 30,
      }

    client = main.app.test_client()
    response = client.post(route, json=payload)
    expected_status = 400 if route == "/run_code" and exit_code else 200
    assert response.status_code == expected_status
    assert "completed" in response.get_data(as_text=True)
    assert not main.sandbox_lock.locked()
    if lose_coverage:
      assert main._temporary_events.fd is None
      assert artifact.exists()
      assert root_artifact.exists()
      assert (self.repo / "moved" / "probe.log").exists()
      return
    assert not artifact.exists()
    assert not nested_artifact.exists()
    assert not root_artifact.exists()
    if route == "/shell":
      tracking_inode = self.tracking.stat().st_ino
      response = client.post(route, json=payload)
      assert response.status_code == expected_status
      assert self.tracking.stat().st_ino == tracking_inode
      assert not artifact.exists()
      assert not nested_artifact.exists()
      assert not root_artifact.exists()

  def test_cleans_on_handler_exception_before_unlocking(self, monkeypatch):
    self._capture()
    artifact = self.roots[0] / "probe.log"
    root_artifact = self.root / "probe.log"
    real_cleanup = main._cleanup_temporary_entries

    def cleanup_while_locked():
      assert main.sandbox_lock.locked()
      real_cleanup()

    monkeypatch.setattr(main, "_cleanup_temporary_entries", cleanup_while_locked)

    @main.lock_sandbox()
    def failing_handler():
      artifact.write_text("created")
      root_artifact.write_text("created")
      raise RuntimeError("handler failed")

    with pytest.raises(RuntimeError, match="handler failed"):
      failing_handler()

    assert not artifact.exists()
    assert not root_artifact.exists()
    assert not main.sandbox_lock.locked()

  @pytest.mark.parametrize("route", ["/shell", "/run_code"])
  @pytest.mark.parametrize("exit_code", [0, 7, None])
  @pytest.mark.parametrize("double_fork", [False, True])
  def test_stops_detached_writer_before_filesystem_cleanup(
    self, route, exit_code, double_fork, linux_subreaper, monkeypatch
  ):
    setup = self.roots[0] / "setup_cache"
    setup.mkdir()
    self._capture()
    artifact = setup / "probe.log"
    pid_file = self.repo / "worker.pid"
    worker_code = (
      "import os, time\n"
      "from pathlib import Path\n"
      # The intermediate parents exit before the writer publishes its PID.
      + (
        "if os.fork(): os._exit(0)\nos.setsid()\nif os.fork(): os._exit(0)\n"
        if double_fork
        else ""
      )
      + f"artifact = Path({str(artifact)!r})\n"
      "artifact.write_text('created')\n"
      f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
      "while True:\n"
      "    artifact.write_text('created')\n"
      "    time.sleep(0.01)\n"
    )
    script = (
      "import subprocess, sys, time\n"
      "from pathlib import Path\n"
      f"worker = subprocess.Popen([sys.executable, '-c', {worker_code!r}],\n"
      "    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
      f"while not Path({str(pid_file)!r}).exists():\n"
      "    if worker.poll() is not None and worker.returncode != 0:\n"
      "        raise RuntimeError('Worker failed to start')\n"
      "    time.sleep(0.01)\n"
      "print('completed', flush=True)\n"
      + (
        "time.sleep(60)\n" if exit_code is None else f"raise SystemExit({exit_code})\n"
      )
    )
    helper = subprocess.Popen(
      [sys.executable, "-c", "import time; time.sleep(60)"],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )
    real_cleanup = main._cleanup_temporary_entries
    checked = []

    def cleanup_after_writer_stops():
      assert main.sandbox_lock.locked()
      pid = int(pid_file.read_text())
      with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
      checked.append(pid)
      real_cleanup()

    monkeypatch.setattr(main, "_cleanup_temporary_entries", cleanup_after_writer_stops)
    payload = (
      {
        "cmd": f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
        "cwd": ".",
        "code_files": {},
      }
      if route == "/shell"
      else {"code": {"probe.py": script}, "evaluation_script": "probe.py"}
    )
    payload["timeout"] = 1.0 if exit_code is None else 30
    try:
      response = main.app.test_client().post(route, json=payload)
      failed_evaluation = route == "/run_code" and exit_code != 0
      assert response.status_code == (400 if failed_evaluation else 200)
      response_text = response.get_data(as_text=True)
      assert ("timed" if exit_code is None else "completed") in response_text
      assert checked
      assert not artifact.exists()
      assert helper.poll() is None
      assert not common_tools.is_contaminated()
    finally:
      helper.kill()
      helper.wait()
      # Keep a failing test from leaving its detached writer in the test container.
      if pid_file.exists():
        pid = int(pid_file.read_text())
        try:
          os.kill(pid, signal.SIGKILL)
          os.waitpid(pid, 0)
        except (ProcessLookupError, ChildProcessError):
          pass

  def test_excludes_workspace_backup_and_git_trees(self, monkeypatch):
    protected = (
      self.roots[0] / "workspace",
      self.roots[0] / "backup",
      self.tracking,
      pathlib.Path(main.REQUEST_TRACKING_GIT_DIR),
      pathlib.Path(main.SCRATCH_TRACKING_GIT_DIR),
    )
    for path in protected:
      path.mkdir()
      (path / "nested").mkdir()
    monkeypatch.setattr(main, "REPO_DIR", str(protected[0]))
    monkeypatch.setattr(main, "BACKUP_DIR", str(protected[1]))
    identities = {(path.stat().st_dev, path.stat().st_ino) for path in protected}
    real_scandir = os.scandir

    def avoid_protected_trees(path):
      info = os.fstat(path) if isinstance(path, int) else os.stat(path)
      assert (info.st_dev, info.st_ino) not in identities
      return real_scandir(path)

    monkeypatch.setattr(main.os, "scandir", avoid_protected_trees)
    self._capture()
    for path in protected:
      (path / "nested" / "new.txt").write_text("preserved")
    (self.roots[0] / "probe.log").write_text("remove")

    main._cleanup_temporary_entries()

    for path in protected:
      assert (path / "nested" / "new.txt").read_text() == "preserved"
    assert not (self.roots[0] / "probe.log").exists()

  def test_blocks_symlinked_ancestor_even_when_leaf_inode_matches(self, monkeypatch):
    setup = self.roots[0] / "setup"
    leaf = setup / "leaf"
    leaf.mkdir(parents=True)
    self._capture()
    leaf.rename(self.repo / "leaf")
    setup.rmdir()
    setup.symlink_to(self.repo, target_is_directory=True)
    protected = self.repo / "leaf" / "new.txt"
    protected.write_text("preserved")
    # Select the original leaf directly to exercise identity checks even when
    # its ancestor has changed since the event was collected.
    monkeypatch.setattr(main._temporary_events, "changed_paths", lambda: {leaf})

    main._cleanup_temporary_entries()

    assert protected.read_text() == "preserved"

  def test_event_initialization_error_propagates(self, monkeypatch):
    libc = main.ctypes.CDLL(None, use_errno=True)

    def failed_init(flags):
      main.ctypes.set_errno(errno.EMFILE)
      return -1

    monkeypatch.setattr(libc, "inotify_init1", failed_init)
    monkeypatch.setattr(main.ctypes, "CDLL", lambda *args, **kwargs: libc)

    with pytest.raises(OSError) as error:
      self._capture()

    assert error.value.errno == errno.EMFILE
    assert main._temporary_events is None

  @pytest.mark.parametrize("failure", ["watch", "inventory", "missing_inventory"])
  def test_startup_errors_close_watcher_and_propagate(self, monkeypatch, failure):
    nested = self.roots[0] / "cache" / "nested"
    nested.mkdir(parents=True)
    events = main._TemporaryDirectoryEvents.open()
    original_fd = events.fd
    monkeypatch.setattr(
      main._TemporaryDirectoryEvents, "open", classmethod(lambda cls: events)
    )

    if failure == "watch":

      def failed_watch(*args):
        main.ctypes.set_errno(errno.ENOSPC)
        return -1

      monkeypatch.setattr(events, "_add_watch", failed_watch)
      expected_errno = errno.ENOSPC
    else:
      # Fail after this directory's watch was installed, before its inventory
      # was published. A read failure must never leave that partial stream live.
      target = self.roots[0] if failure == "missing_inventory" else nested
      identity = (target.stat().st_dev, target.stat().st_ino)
      real_scandir = os.scandir
      expected_errno = errno.ENOENT if failure == "missing_inventory" else errno.EACCES

      def unreadable(path):
        info = os.fstat(path)
        if (info.st_dev, info.st_ino) == identity:
          assert target in events.paths.values()
          raise OSError(expected_errno, "Inventory failed", str(target))
        return real_scandir(path)

      monkeypatch.setattr(main.os, "scandir", unreadable)

    with pytest.raises(OSError) as error:
      self._capture()

    assert error.value.errno == expected_errno
    assert main._temporary_events is None
    with pytest.raises(OSError) as closed_fd:
      os.fstat(original_fd)
    assert closed_fd.value.errno == errno.EBADF

  @pytest.mark.parametrize("location", ["root", "temporary", "image"])
  def test_scans_only_event_dirty_directories(self, monkeypatch, location):
    changed = {
      "root": self.root,
      "temporary": self.roots[0] / "cache" / "nested",
      "image": self.root / "opt" / "runtime" / "lib",
    }[location]
    quiet = self.root / "usr" / "local" / "untouched_source"
    changed.mkdir(parents=True, exist_ok=True)
    quiet.mkdir(parents=True)
    (quiet / "source.py").write_text("preserved")
    self._capture()
    assert main._temporary_events.fd is not None
    changed_info = changed.stat()
    real_scandir = os.scandir
    scanned = []

    def only_changed_directory(path):
      info = os.fstat(path) if isinstance(path, int) else os.stat(path)
      identity = (info.st_dev, info.st_ino)
      assert identity == (changed_info.st_dev, changed_info.st_ino)
      scanned.append(identity)
      return real_scandir(path)

    monkeypatch.setattr(main.os, "scandir", only_changed_directory)

    def no_watch_recovery(*args, **kwargs):
      pytest.fail("Cleanup registered a watch after startup.")

    monkeypatch.setattr(main._temporary_events, "watch", no_watch_recovery)
    main._cleanup_temporary_entries()
    assert scanned == []
    (changed / "probe.log").write_text("remove")

    main._cleanup_temporary_entries()

    assert not (changed / "probe.log").exists()
    assert len(scanned) == 1
    scanned.clear()
    main._cleanup_temporary_entries()
    assert scanned == []

  def test_detects_tree_renamed_into_setup_directory(self):
    setup = self.roots[0] / "cache"
    setup.mkdir()
    self._capture()
    incoming = self.repo / "incoming"
    (incoming / "nested").mkdir(parents=True)
    (incoming / "nested" / "result").write_text("remove")
    incoming.rename(setup / "incoming")

    main._cleanup_temporary_entries()

    assert not (setup / "incoming").exists()

  @pytest.mark.parametrize("failure", ["overflow", "deleted", "read"])
  def test_coverage_loss_stops_cleanup_without_scans_or_recovery(
    self, monkeypatch, caplog, failure
  ):
    setup = self.roots[0] / "cache"
    setup.mkdir()
    self._capture()
    events = main._temporary_events
    artifact = self.roots[1] / "probe.log"
    artifact.write_text("left for retirement")
    if failure == "deleted":
      setup.rmdir()
    else:
      real_read = os.read

      def failed_stream(fd, count):
        if fd != events.fd:
          return real_read(fd, count)
        if failure == "read":
          raise OSError(errno.EIO, "Event read failed")
        real_read(fd, count)
        return main._IN_EVENT.pack(-1, main._IN_Q_OVERFLOW, 0, 0)

      monkeypatch.setattr(main.os, "read", failed_stream)

    def no_directory_work(*args, **kwargs):
      pytest.fail("Lost coverage triggered a scan or watch recovery.")

    monkeypatch.setattr(main, "_open_temporary_directory", no_directory_work)
    monkeypatch.setattr(events, "watch", no_directory_work)
    with caplog.at_level(logging.WARNING):
      main._cleanup_temporary_entries()
      main._cleanup_temporary_entries()

    assert events.fd is None
    assert artifact.exists()
    assert "stopping cleanup" in caplog.text
    reason = main._filesystem_contamination_reason
    assert {
      "overflow": "queue overflowed",
      "deleted": "watch was lost",
      "read": "tracking failed",
    }[failure] in reason

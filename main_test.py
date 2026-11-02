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


class TestTemporaryEntryCleanup:
  """Exercise cleanup in private roots, including both HTTP execution paths."""

  @pytest.fixture(autouse=True)
  def _sandbox(self, monkeypatch, tmp_path):
    self.roots = tuple(tmp_path / name for name in ("tmp", "var_tmp", "shm"))
    for root in self.roots:
      root.mkdir()
    self.repo = tmp_path / "repo"
    self.backup = tmp_path / "backup"
    self.repo.mkdir()
    self.backup.mkdir()
    self.tracking = self.roots[0] / ".agent_git"
    monkeypatch.setattr(main, "_TEMPORARY_DIRS", self.roots)
    monkeypatch.setattr(main, "_temporary_baseline", {})
    monkeypatch.setattr(main, "_temporary_events", None)
    monkeypatch.setattr(main, "REPO_DIR", f"{self.repo}/")
    monkeypatch.setattr(main, "BACKUP_DIR", f"{self.backup}/")
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

  @pytest.mark.parametrize("replacement", ["symlink", "directory"])
  def test_skips_roots_replaced_after_capture(self, replacement, caplog):
    self._capture()
    root = self.roots[0]
    root.rename(root.with_name("old_tmp"))
    if replacement == "symlink":
      root.symlink_to(self.repo, target_is_directory=True)
    else:
      root.mkdir()
    protected = root / "keep.txt"
    protected.write_text("keep")
    (self.roots[1] / "new.log").write_text("new")

    with caplog.at_level(logging.WARNING):
      main._cleanup_temporary_entries()

    assert protected.read_text() == "keep"
    assert not (self.roots[1] / "new.log").exists()
    assert caplog.records
    if sys.platform == "linux":
      assert main._temporary_events.fd is not None
    (self.roots[1] / "later.log").write_text("remove after watch loss")
    main._cleanup_temporary_entries()
    assert not (self.roots[1] / "later.log").exists()

  def test_logs_removal_failure_and_continues(self, monkeypatch, caplog):
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
    assert "Could not remove temporary entry" in caplog.text
    monkeypatch.setattr(main.os, "unlink", real_unlink)
    main._cleanup_temporary_entries()
    assert not blocked.exists()

  @pytest.mark.parametrize("route", ["/shell", "/run_code"])
  @pytest.mark.parametrize("exit_code", [0, 7])
  def test_cleans_after_execution_and_preserves_response(self, route, exit_code):
    nested = self.roots[0] / "setup_cache" / "nested"
    nested.mkdir(parents=True)
    self._capture()
    artifact = self.roots[0] / "probe.log"
    nested_artifact = nested / "probe.log"
    script = (
      "from pathlib import Path\n"
      f"Path({str(artifact)!r}).write_text('created')\n"
      f"Path({str(nested_artifact)!r}).write_text('created')\n"
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
    assert not artifact.exists()
    assert not nested_artifact.exists()
    if route == "/shell":
      tracking_inode = self.tracking.stat().st_ino
      response = client.post(route, json=payload)
      assert response.status_code == expected_status
      assert self.tracking.stat().st_ino == tracking_inode
      assert not artifact.exists()
      assert not nested_artifact.exists()

  def test_cleans_on_handler_exception_before_unlocking(self, monkeypatch):
    self._capture()
    artifact = self.roots[0] / "probe.log"
    real_cleanup = main._cleanup_temporary_entries

    def cleanup_while_locked():
      assert main.sandbox_lock.locked()
      real_cleanup()

    monkeypatch.setattr(main, "_cleanup_temporary_entries", cleanup_while_locked)

    @main.lock_sandbox()
    def failing_handler():
      artifact.write_text("created")
      raise RuntimeError("handler failed")

    with pytest.raises(RuntimeError, match="handler failed"):
      failing_handler()

    assert not artifact.exists()
    assert not main.sandbox_lock.locked()

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

  def test_blocks_symlinked_ancestor_even_when_leaf_inode_matches(self):
    setup = self.roots[0] / "setup"
    leaf = setup / "leaf"
    leaf.mkdir(parents=True)
    self._capture()
    leaf.rename(self.repo / "leaf")
    setup.rmdir()
    setup.symlink_to(self.repo, target_is_directory=True)
    protected = self.repo / "leaf" / "new.txt"
    protected.write_text("preserved")

    main._cleanup_temporary_entries()

    assert protected.read_text() == "preserved"

  def test_falls_back_when_event_tracking_is_unavailable(self, monkeypatch):
    nested = self.roots[0] / "cache" / "nested"
    nested.mkdir(parents=True)
    monkeypatch.setattr(
      main._TemporaryDirectoryEvents, "open", classmethod(lambda cls: None)
    )
    self._capture()
    (nested / "probe.log").write_text("remove")

    main._cleanup_temporary_entries()

    assert not (nested / "probe.log").exists()

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_scans_only_event_dirty_directories(self, monkeypatch):
    changed = self.roots[0] / "cache" / "nested"
    quiet = self.roots[0] / "untouched_source"
    changed.mkdir(parents=True)
    quiet.mkdir()
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
    main._cleanup_temporary_entries()
    assert scanned == []
    (changed / "probe.log").write_text("remove")

    main._cleanup_temporary_entries()

    assert not (changed / "probe.log").exists()
    assert len(scanned) == 1
    scanned.clear()
    main._cleanup_temporary_entries()
    assert scanned == []

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
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

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_deleted_setup_tree_keeps_other_watches_and_quiet_calls_idle(
    self, monkeypatch
  ):
    removed = self.roots[0] / "setup_cache"
    (removed / "nested").mkdir(parents=True)
    quiet = self.roots[1] / "quiet"
    quiet.mkdir()
    self._capture()
    baseline = main._temporary_baseline
    events = main._temporary_events
    original_fd = events.fd
    quiet_watch = next(wd for wd, path in events.paths.items() if path == quiet)
    shutil.rmtree(removed)
    (quiet / "probe.log").write_text("remove during recovery")

    main._cleanup_temporary_entries()

    assert events.fd == original_fd
    assert events.paths[quiet_watch] == quiet
    assert main._temporary_baseline is baseline
    assert not (quiet / "probe.log").exists()
    real_open = main._open_temporary_directory

    def no_directory_work(*args, **kwargs):
      pytest.fail("An idle call revisited setup directories after watch recovery.")

    monkeypatch.setattr(main, "_open_temporary_directory", no_directory_work)
    main._cleanup_temporary_entries()
    monkeypatch.setattr(main, "_open_temporary_directory", real_open)
    (quiet / "next.log").write_text("remove using the surviving watch")
    main._cleanup_temporary_entries()
    assert not (quiet / "next.log").exists()

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_moved_setup_tree_returns_to_original_baseline(self):
    setup = self.roots[0] / "cache"
    nested = setup / "nested"
    nested.mkdir(parents=True)
    (nested / "setup.txt").write_text("preserved")
    self._capture()
    events = main._temporary_events
    original_fd = events.fd
    moved = self.repo / "moved_cache"
    setup.rename(moved)
    artifact = moved / "nested" / "probe.log"
    artifact.write_text("created while displaced")

    main._cleanup_temporary_entries()

    assert events.fd == original_fd
    assert artifact.exists()
    moved.rename(setup)
    main._cleanup_temporary_entries()
    assert events.fd == original_fd
    assert (nested / "setup.txt").read_text() == "preserved"
    assert not (nested / "probe.log").exists()
    (nested / "next.log").write_text("remove after returning")
    main._cleanup_temporary_entries()
    assert not (nested / "next.log").exists()

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_ignored_watch_is_rearmed_without_accepting_new_entries(self):
    setup = self.roots[0] / "cache"
    setup.mkdir()
    (setup / "setup.txt").write_text("preserved")
    self._capture()
    events = main._temporary_events
    original_fd = events.fd
    descriptor = next(wd for wd, path in events.paths.items() if path == setup)
    libc = main.ctypes.CDLL(None, use_errno=True)
    libc.inotify_rm_watch.argtypes = [main.ctypes.c_int, main.ctypes.c_int]
    libc.inotify_rm_watch.restype = main.ctypes.c_int
    assert libc.inotify_rm_watch(events.fd, descriptor) == 0
    (setup / "probe.log").write_text("created without a watch")

    main._cleanup_temporary_entries()

    assert events.fd == original_fd
    assert setup in events.paths.values()
    assert (setup / "setup.txt").read_text() == "preserved"
    assert not (setup / "probe.log").exists()
    assert "probe.log" not in main._temporary_baseline[setup].entries
    (setup / "next.log").write_text("remove using recovered watch")
    main._cleanup_temporary_entries()
    assert not (setup / "next.log").exists()

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_recovery_watch_failure_keeps_scan_cleanup_available(
    self, monkeypatch, caplog
  ):
    setup = self.roots[0] / "cache"
    setup.mkdir()
    self._capture()
    setup.rmdir()
    (self.roots[1] / "probe.log").write_text("remove despite recovery failure")

    def failed_watch(events, path, directory_fd):
      raise OSError(errno.ENOSPC, "No watches available")

    monkeypatch.setattr(main._TemporaryDirectoryEvents, "watch", failed_watch)
    with caplog.at_level(logging.WARNING):
      main._cleanup_temporary_entries()

    assert main._temporary_events.fd is None
    assert not (self.roots[1] / "probe.log").exists()
    assert "Could not recover temporary directory watch" in caplog.text

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_overflow_scans_all_directories_then_resumes_events(
    self, monkeypatch, caplog
  ):
    quiet = self.roots[0] / "quiet"
    quiet.mkdir()
    self._capture()
    events = main._temporary_events
    (self.roots[1] / "probe.log").write_text("remove")
    (quiet / "probe.log").write_text("remove without its creation event")
    real_read = os.read
    real_scandir = os.scandir
    scanned = []

    def overflow_record(fd, count):
      data = real_read(fd, count)
      if fd == events.fd:
        return main._IN_EVENT.pack(-1, main._IN_Q_OVERFLOW, 0, 0)
      return data

    def record_scans(path):
      info = os.fstat(path) if isinstance(path, int) else os.stat(path)
      scanned.append((info.st_dev, info.st_ino))
      return real_scandir(path)

    monkeypatch.setattr(main.os, "read", overflow_record)
    monkeypatch.setattr(main.os, "scandir", record_scans)
    with caplog.at_level(logging.WARNING):
      main._cleanup_temporary_entries()

    assert not (self.roots[1] / "probe.log").exists()
    assert not (quiet / "probe.log").exists()
    assert (quiet.stat().st_dev, quiet.stat().st_ino) in scanned
    assert "queue overflowed" in caplog.text
    assert events.fd is not None
    monkeypatch.setattr(main.os, "read", real_read)
    scanned.clear()
    main._cleanup_temporary_entries()
    assert scanned == []

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_init_failure_keeps_nested_cleanup_available(self, monkeypatch, caplog):
    nested = self.roots[0] / "cache" / "nested"
    nested.mkdir(parents=True)

    def unavailable_inotify(*args, **kwargs):
      raise OSError(errno.ENOSYS, "Inotify is unavailable")

    monkeypatch.setattr(main.ctypes, "CDLL", unavailable_inotify)
    self._capture()
    assert main._temporary_events is None
    (nested / "probe.log").write_text("remove")

    main._cleanup_temporary_entries()

    assert not (nested / "probe.log").exists()
    assert "will use directory scans" in caplog.text

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_watch_failure_keeps_nested_cleanup_available(self, monkeypatch, caplog):
    nested = self.roots[0] / "cache" / "nested"
    nested.mkdir(parents=True)

    def unavailable_watch(events, path, directory_fd):
      if events.fd is not None:
        raise OSError(errno.ENOSPC, "No watches available")

    monkeypatch.setattr(main._TemporaryDirectoryEvents, "watch", unavailable_watch)
    self._capture()
    assert main._temporary_events.fd is None
    (nested / "probe.log").write_text("remove")

    main._cleanup_temporary_entries()

    assert not (nested / "probe.log").exists()
    assert "Could not watch temporary directory" in caplog.text

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_read_failure_falls_back_to_nested_cleanup(self, monkeypatch, caplog):
    nested = self.roots[0] / "cache" / "nested"
    nested.mkdir(parents=True)
    self._capture()
    events = main._temporary_events
    (nested / "probe.log").write_text("remove")
    real_read = os.read

    def failed_read(fd, count):
      if fd == events.fd:
        raise OSError(errno.EIO, "Event read failed")
      return real_read(fd, count)

    monkeypatch.setattr(main.os, "read", failed_read)
    with caplog.at_level(logging.WARNING):
      main._cleanup_temporary_entries()

    assert not (nested / "probe.log").exists()
    assert events.fd is None
    assert "using directory scans" in caplog.text

  @pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux inotify")
  def test_keeps_events_arriving_during_cleanup(self, monkeypatch):
    setup = self.roots[0] / "cache"
    setup.mkdir()
    self._capture()
    first = setup / "first.tmp"
    late = setup / "late.tmp"
    first.write_text("remove")
    real_scandir = os.scandir
    real_unlink = os.unlink

    @contextlib.contextmanager
    def snapshot_scandir(path):
      with real_scandir(path) as entries:
        snapshot = list(entries)
      yield iter(snapshot)

    def create_during_removal(path, **kwargs):
      result = real_unlink(path, **kwargs)
      if path == first.name:
        late.write_text("created during cleanup")
      return result

    monkeypatch.setattr(main.os, "scandir", snapshot_scandir)
    monkeypatch.setattr(main.os, "unlink", create_during_removal)
    main._cleanup_temporary_entries()
    assert late.exists()

    main._cleanup_temporary_entries()

    assert not late.exists()

"""Tests for the main module."""

import contextlib
import json
import os
import pathlib
import shlex
import signal
import sys
import time

import pytest

import common_tools
import filesystem_cleanup
import main
from conftest import wait_until_dead


@pytest.fixture(autouse=True)
def _isolated_filesystem(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
):
  """Never inventory or clean the development machine's real filesystem."""
  monkeypatch.setattr(main, "FILESYSTEM_ROOT", tmp_path)
  common_tools._contaminated.clear()
  yield
  common_tools._contaminated.clear()


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
    monkeypatch.setattr(main.subprocess, "run", lambda *a, **k: None)
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
  def test_rejects_code_file_paths_outside_repo(
    self, content: str | None
  ) -> None:
    """Overlay writes and deletions cannot escape through parent traversal."""
    outside = self.repo.parent / "outside.txt"
    outside.write_text("keep me")

    with pytest.raises(ValueError, match="Path escapes repo directory"):
      main.execute_python_function(
        {"../outside.txt": content}, [], 30, "unused.py"
      )

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
      main.execute_python_function(
        {"escape/outside.txt": None}, [], 30, "unused.py"
      )

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
      main.execute_shell_command(
        "true", str(self.repo), {"../outside.txt": None}, 30
      )

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
    pid_file = self.repo / "job.pid"
    cmd = f"sleep 30 & echo $! > {pid_file}; sleep 30"

    # when it is run with a short timeout.
    output = self._run(cmd, timeout=1)

    # then the timeout is reported and the backgrounded job was killed with it.
    assert output == "Error: command timed out after 1s"
    job_pid = int(pid_file.read_text())
    assert wait_until_dead(job_pid), (
      f"backgrounded job {job_pid} survived the timeout"
    )

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
    pid_file = self.repo / "escaper.pid"
    cmd = (
      f"{sys.executable} -c "
      "'import subprocess, sys, os; "
      "child = subprocess.Popen("
      "  [sys.executable, \"-c\", \"import time; time.sleep(30)\"], "
      "  preexec_fn=os.setsid); "
      f'open("{pid_file}", "w").write(str(child.pid)); '
      "print(\"partial output\")'"
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

  def test_schedules_a_restart_once_the_pod_is_contaminated(
    self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    """
    A survivor triggers a container restart.
    """
    # given a contaminated pod, and a stand-in for the timer so the exit never
    # runs here.
    restarts: list[float] = []
    monkeypatch.setattr(main.threading, "Timer", lambda delay, fn: _MockTimer(delay, restarts))
    common_tools.mark_contaminated()

    # when a response goes out.
    main._restart_if_contaminated(_Response())

    # then a restart was scheduled at the grace delay.
    assert restarts == [main._RESTART_GRACE]


class TestShellChangeTracking:
  """
  Tests the tracking repo that reports what a shell command changed.

  The repo lives for the pod, not for the call: re-creating it would force
  `git add -A` to re-hash the whole work tree. These tests pin the behaviour the
  surviving index has to keep -- the baseline is the restored work tree, not
  whatever the previous call left behind.
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
      f'{sys.executable} -c '
      '\'open("blob.bin", "wb").write(b"\\xff\\xfe\\x00binary")\''
    )
    files = self._changed(f"{write_blob}; echo real > kept.txt")

    # then the text file comes back and the binary is left out.
    assert files == {"kept.txt": "real\n"}

  def test_changes_do_not_persist_into_the_next_call(self) -> None:
    """
    The surviving index tracks the restored work tree, so a call is never charged
    for what the previous call did.
    """
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


class TestExternalCleanup:
  """Exercise external cleanup through real shell and evaluation requests."""

  @pytest.fixture(autouse=True)
  def _sandbox(
    self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
  ) -> None:
    self.root = tmp_path
    self.repo = tmp_path / "app"
    self.backup = tmp_path / "shared" / "repo"
    self.repo.mkdir()
    self.backup.mkdir(parents=True)
    monkeypatch.setattr(main, "REPO_DIR", f"{self.repo}/")
    monkeypatch.setattr(main, "BACKUP_DIR", f"{self.backup}/")
    monkeypatch.setattr(main, "TRACKING_GIT_DIR", str(tmp_path / ".agent_git"))
    monkeypatch.setenv("PYTHON_EXECUTABLE", sys.executable)

  def _shell(self, cmd: str, timeout: float = 30):
    return main.app.test_client().post(
      "/shell",
      json={"cmd": cmd, "cwd": ".", "code_files": {}, "timeout": timeout},
    )

  def _evaluate(self, script: str, timeout: float = 30):
    return main.app.test_client().post(
      "/run_code",
      json={
        "code": {"evaluator.py": script},
        "timeout": timeout,
        "args": [],
      },
    )

  @pytest.mark.parametrize(
    ("ending", "timeout", "status", "output"),
    [
      ("print('result')", 30, 200, "result"),
      ("raise SystemExit(7)", 30, 400, "non-zero exit code (7)"),
      ("time.sleep(30)", 1, 400, "Evaluation timed-out after 1.0 seconds."),
    ],
  )
  def test_evaluation_cleans_up_after_success_failure_and_timeout(
    self, ending: str, timeout: float, status: int, output: str,
  ) -> None:
    """Evaluator artifacts are removed while setup data and repo files survive."""
    outside = self.root / "scratch"
    outside.mkdir()
    (outside / "existing.txt").write_text("setup data")
    response = self._evaluate(
      "from pathlib import Path\n"
      "import time\n"
      f"outside = Path({str(outside)!r})\n"
      "(outside / 'new-directory').mkdir()\n"
      "(outside / 'new-directory' / 'data').write_text('scratch')\n"
      "Path('result.txt').write_text('candidate')\n"
      f"{ending}\n",
      timeout,
    )

    assert response.status_code == status
    assert output in response.get_data(as_text=True)
    assert not (outside / "new-directory").exists()
    assert (outside / "existing.txt").read_text() == "setup data"
    assert (self.repo / "result.txt").read_text() == "candidate"

  def test_checkpoint_is_read_before_external_cleanup(self) -> None:
    """A checkpoint pointing at a new external file can be read before removal."""
    outside = self.root / "checkpoint.json"
    script = (
      "from pathlib import Path\n"
      f"checkpoint = Path({str(outside)!r})\n"
      "checkpoint.write_text('{\"score\": 1}')\n"
      "Path('checkpoint.json').symlink_to(checkpoint)\n"
      "raise SystemExit(7)\n"
    )

    with main.app.app_context():
      response, status = main.execute_python_function(
        {"evaluator.py": script}, [], 30, "evaluator.py",
      )

    assert status == 200
    assert response.get_json()["data"] == {
      "output": '{"score": 1}', "metainfo": "Checkpoint",
    }
    assert not outside.exists()

  @pytest.mark.parametrize(
    ("ending", "timeout", "expected_output"),
    [
      ("echo done", 30, "done"),
      ("exit 7", 30, "Exit code: 7"),
      ("sleep 30", 1, "Error: command timed out after 1.0s"),
    ],
  )
  def test_removes_new_paths_after_success_failure_and_timeout(
    self, ending: str, timeout: float, expected_output: str,
  ) -> None:
    """External artifacts vanish before the response; repo edits are returned."""
    locations = [
      self.root / "tmp",
      self.root / "home" / "worker" / ".cache",
      self.root / "opt",
      self.root / "shared",
      self.root / "dev" / "shm",
    ]
    for location in locations:
      location.mkdir(parents=True, exist_ok=True)
      (location / "existing.txt").write_text("setup data")
    new_tree = self.root / "new tree"
    quoted_locations = " ".join(shlex.quote(str(path)) for path in locations)
    response = self._shell(
      f"for directory in {quoted_locations}; do "
      'echo scratch > "$directory/new.txt"; '
      'mkdir -p "$directory/new-directory/nested"; '
      'echo nested > "$directory/new-directory/nested/data"; '
      "done; "
      f"mkdir -p {shlex.quote(str(new_tree))}; "
      "echo candidate > candidate.txt; "
      f"{ending}",
      timeout,
    )

    assert response.status_code == 200
    result = response.get_json()["result"]
    assert expected_output in result["output"]
    assert result["files"] == {"candidate.txt": "candidate\n"}
    for location in locations:
      assert not (location / "new.txt").exists()
      assert not (location / "new-directory").exists()
      assert (location / "existing.txt").read_text() == "setup data"
    assert not new_tree.exists()
    assert (self.root / ".agent_git" / "HEAD").exists()

  def test_new_symlinks_are_unlinked_without_touching_targets(self) -> None:
    """Neither dangling links nor directory links are traversed during cleanup."""
    outside = self.root / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (self.backup / "keep.txt").write_text("candidate")
    existing_link = self.root / "existing-link"
    existing_link.symlink_to(outside, target_is_directory=True)
    existing_dangling = self.root / "existing-dangling"
    existing_dangling.symlink_to(self.root / "missing")
    new_link = self.root / "new-link"
    new_dangling = self.root / "new-dangling"
    response = self._shell(
      f"ln -s {shlex.quote(str(self.repo))} {shlex.quote(str(new_link))}; "
      f"ln -s missing {shlex.quote(str(new_dangling))}"
    )

    assert response.status_code == 200
    assert not new_link.is_symlink()
    assert not new_dangling.is_symlink()
    assert existing_link.is_symlink()
    assert existing_dangling.is_symlink()
    assert (outside / "keep.txt").read_text() == "keep"
    assert (self.repo / "keep.txt").read_text() == "candidate"

  def test_kernel_paths_are_excluded(self) -> None:
    """Kernel-managed entries are left alone, while /dev/shm is cleaned."""
    paths = [self.root / name for name in ("proc", "sys", "dev", "dev/shm")]
    for path in paths:
      path.mkdir(parents=True, exist_ok=True)
    quoted_paths = " ".join(shlex.quote(str(path)) for path in paths)

    response = self._shell(
      f"for directory in {quoted_paths}; do touch \"$directory/new\"; done"
    )

    assert response.status_code == 200
    assert all((path / "new").exists() for path in paths[:-1])
    assert not (paths[-1] / "new").exists()

  def test_cleans_up_if_collecting_repo_changes_fails(
    self, monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """An exception after command execution still removes external artifacts."""
    outside = self.root / "new.txt"

    def fail_collection():
      raise RuntimeError("Could not collect changes")

    monkeypatch.setattr(main, "get_changed_files", fail_collection)
    response = self._shell(f"touch {shlex.quote(str(outside))}")

    assert response.status_code == 500
    assert not outside.exists()

  @pytest.mark.parametrize("route", ["/shell", "/run_code"])
  def test_cleanup_failure_restarts_and_rejects_further_execution(
    self, monkeypatch: pytest.MonkeyPatch, route: str,
  ) -> None:
    """A failed cleanup cannot be followed by another shell or evaluation."""
    outside = self.root / "new-directory"
    original_rmtree = filesystem_cleanup.shutil.rmtree

    def fail_cleanup(path, *args, **kwargs):
      if pathlib.Path(path) == outside:
        raise PermissionError("Cannot remove external directory")
      return original_rmtree(path, *args, **kwargs)

    restarts: list[float] = []
    monkeypatch.setattr(filesystem_cleanup.shutil, "rmtree", fail_cleanup)
    monkeypatch.setattr(
      main.threading, "Timer", lambda delay, fn: _MockTimer(delay, restarts),
    )
    if route == "/shell":
      response = self._shell(f"mkdir {shlex.quote(str(outside))}")
    else:
      response = self._evaluate(
        f"from pathlib import Path\nPath({str(outside)!r}).mkdir()\nprint('done')"
      )

    assert response.status_code == 500
    assert common_tools.is_contaminated()
    assert restarts == [main._RESTART_GRACE]
    assert self._shell("echo should-not-run > unexpected.txt").status_code == 503
    response = main.app.test_client().post(
      "/run_code", json={"code": {}, "timeout": 30},
    )
    assert response.status_code == 503
    assert not (self.repo / "unexpected.txt").exists()

  def test_incomplete_inventory_does_not_run_command(
    self, monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Unreadable directories must not yield an unsafe, partial baseline."""
    unreadable = self.root / "unreadable"
    unreadable.mkdir()
    (unreadable / "existing.txt").write_text("keep")
    original_scandir = filesystem_cleanup.os.scandir

    def fail_inventory(path):
      if pathlib.Path(path) == unreadable:
        raise PermissionError("Cannot inventory directory")
      return original_scandir(path)

    monkeypatch.setattr(filesystem_cleanup.os, "scandir", fail_inventory)
    response = self._shell("echo should-not-run > unexpected.txt")

    assert response.status_code == 500
    assert not (self.repo / "unexpected.txt").exists()
    assert (unreadable / "existing.txt").read_text() == "keep"
    assert not common_tools.is_contaminated()

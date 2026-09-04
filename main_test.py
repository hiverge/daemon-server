"""Tests for the main module."""

import contextlib
import json
import os
import pathlib
import signal
import sys
import time

import pytest

import common_tools
import main
from conftest import wait_until_dead


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
    pid_file = tmp_path / "job.pid"
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
    pid_file = tmp_path / "escaper.pid"
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

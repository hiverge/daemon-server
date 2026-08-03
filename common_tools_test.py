"""Tests for the common_tools module."""

import contextlib
import logging
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

import common_tools
from conftest import wait_until_dead


def _python(code: str) -> list[str]:
  """
  Build a command that runs the given Python source in a subprocess.
  """
  return [sys.executable, "-c", code]


class TestRunCommand:
  """
  Tests for the `run_command` function.
  """

  def test_returns_last_output_line_on_success(self) -> None:
    """
    A command that exits 0 returns only the final line of its stdout.
    """
    # given a command that prints two lines and exits cleanly.
    cmd = _python("print('first line'); print('final line')")

    # when it is run.
    result = common_tools.run_command(cmd)

    # then only the final line is returned.
    assert result == "final line"

  def test_raises_no_output_when_stdout_empty(self) -> None:
    """
    A command that exits 0 without printing anything is reported as producing
    no output rather than crashing.
    """
    # given a command that exits cleanly without printing anything.
    cmd = _python("pass")

    # when it is run, then a no-output error is raised.
    with pytest.raises(
      common_tools.FunctionExecutionError,
      match=r"^Evaluator Format Error: No output was written\.$",
    ):
      common_tools.run_command(cmd)

  def test_raises_exit_code_and_tail_on_nonzero_exit(self) -> None:
    """
    A command that exits non-zero reports its exit code together with the tail
    of both output streams under their own underlined headers.
    """
    # given a command that writes to both streams then exits non-zero.
    cmd = _python(
      "import sys; "
      "print('some earlier noisy output'); "
      "print('evaluator progress on stdout'); "
      "print('evaluator failed on stderr', file=sys.stderr); "
      "sys.exit(2)"
    )

    # when it is run, then the error carries the exit code and both stream tails.
    with pytest.raises(common_tools.FunctionExecutionError) as exc_info:
      common_tools.run_command(cmd)

    assert str(exc_info.value) == (
      "The evaluator returned a non-zero exit code (2) with the following "
      "output:\n\n"
      "stdout (last 20 lines)\n"
      "----------------------\n"
      "some earlier noisy output\n"
      "evaluator progress on stdout\n\n"
      "stderr (last 20 lines)\n"
      "----------------------\n"
      "evaluator failed on stderr"
    )

  def test_reports_stderr_tail_when_stdout_empty_on_nonzero_exit(self) -> None:
    """
    A command that writes only to stderr before failing reports the stderr tail
    and a no-output placeholder for stdout.
    """
    # given a command that writes only to stderr then exits non-zero.
    cmd = _python(
      "import sys; "
      "print('boom on stderr', file=sys.stderr); "
      "sys.exit(1)"
    )

    # when it is run, then stderr carries the tail and stdout shows no output.
    with pytest.raises(common_tools.FunctionExecutionError) as exc_info:
      common_tools.run_command(cmd)

    assert str(exc_info.value) == (
      "The evaluator returned a non-zero exit code (1) with the following "
      "output:\n\n"
      "stdout (last 20 lines)\n"
      "----------------------\n"
      "<No output>\n\n"
      "stderr (last 20 lines)\n"
      "----------------------\n"
      "boom on stderr"
    )

  def test_retains_only_the_tail_of_large_output_on_nonzero_exit(self) -> None:
    """
    A command that prints far more than the retained tail reports only the last
    `MAX_ERROR_OUTPUT_LINES` lines of each stream, bounding memory use.
    """
    # given a command that prints 1000 lines to each stream then exits non-zero.
    cmd = _python(
      "import sys\n"
      "for i in range(1000):\n"
      "    print(f'out{i}')\n"
      "    print(f'err{i}', file=sys.stderr)\n"
      "sys.exit(1)"
    )

    # when it is run, then only the final 20 lines of each stream are reported.
    with pytest.raises(common_tools.FunctionExecutionError) as exc_info:
      common_tools.run_command(cmd)

    expected_stderr_tail = "\n".join(f"err{i}" for i in range(980, 1000))
    expected_stdout_tail = "\n".join(f"out{i}" for i in range(980, 1000))
    assert str(exc_info.value) == (
      "The evaluator returned a non-zero exit code (1) with the following "
      "output:\n\n"
      "stdout (last 20 lines)\n"
      "----------------------\n"
      f"{expected_stdout_tail}\n\n"
      "stderr (last 20 lines)\n"
      "----------------------\n"
      f"{expected_stderr_tail}"
    )

  def test_raises_execution_failed_on_signal(self) -> None:
    """
    A command killed by a signal reports the signal with an execution-failed
    prefix.
    """
    # given a command that dereferences a null pointer to force a SIGSEGV.
    cmd = _python("import ctypes; ctypes.string_at(0)")

    # when it is run, then a SIGSEGV execution-failed error is raised.
    with pytest.raises(
      common_tools.FunctionExecutionError,
      match=r"Terminated by signal 11 \(SIGSEGV\)",
    ):
      common_tools.run_command(cmd)

  def test_raises_execution_failed_on_timeout(self) -> None:
    """
    A command that exceeds the timeout reports an execution-failed timeout.
    """
    # given a command that sleeps well beyond the timeout.
    cmd = _python("import time; time.sleep(30)")

    # when it is run with a short timeout, then a timeout error is raised.
    with pytest.raises(
      common_tools.FunctionExecutionError,
      match=r"Evaluation timed-out after 0\.5 seconds\.",
    ):
      common_tools.run_command(cmd, timeout=0.5)


class TestOrphanedDescendants:
  """
  Tests that a command leaving background processes behind can neither wedge the
  request nor keep running on the pod.

  A descendant inherits the command's stdout/stderr, so the pipes end only once
  every one of them has exited.
  """

  @pytest.fixture(autouse=True)
  def _short_join_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Shorten the reader deadline so these tests stay fast, and clear the
    process-wide contamination flag.
    """
    monkeypatch.setattr(common_tools, "READER_JOIN_TIMEOUT", 1.0)
    common_tools._contaminated.clear()

  def test_returns_promptly_when_orphan_holds_output(self) -> None:
    """
    An evaluator that exits cleanly but leaves a child holding its output fails
    fast, rather than blocking until the child exits.
    """
    # given an evaluator that spawns a long-lived child sharing its stdout/stderr
    # and then exits cleanly itself.
    cmd = _python(
      "import subprocess, sys; "
      "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
      "print('fitness: 1.0')"
    )

    # when it is run, then it fails naming the leaked processes as the cause.
    start = time.monotonic()
    with pytest.raises(
      common_tools.FunctionExecutionError,
      match=r"left background processes still holding its output",
    ):
      common_tools.run_command(cmd, timeout=30)

    # and it returns on the reader deadline, not the child's 60s lifetime.
    assert time.monotonic() - start < 20

  def test_kills_the_evaluator_when_the_run_fails_unexpectedly(
    self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    """
    A failure on a path with no explicit kill still leaves nothing running.
    """
    # given a run that raises straight after the spawn, before anything drains or
    # kills the process group.
    pids: list[int] = []

    def fail(process: subprocess.Popen, *_: object) -> None:
      pids.append(process.pid)
      raise RuntimeError("boom")

    monkeypatch.setattr(common_tools, "start_stream_readers", fail)

    # when it is run, then the failure propagates.
    with pytest.raises(RuntimeError, match="boom"):
      common_tools.run_command(_python("import time; time.sleep(60)"), timeout=30)

    # and the evaluator was killed and reaped, so the pod needs no restart.
    assert wait_until_dead(pids[0])
    assert not common_tools.is_contaminated()

  def test_kills_orphan_that_outlives_a_timed_out_evaluator(
    self, tmp_path: pathlib.Path
  ) -> None:
    """
    Timing out kills the whole process group, not just the evaluator, so no
    descendant is left consuming the pod's resources.
    """
    # given an evaluator that spawns a long-lived child and then hangs itself. The
    # child's pid goes to a file, as a timed-out run reports only the timeout.
    pid_file = tmp_path / "child.pid"
    cmd = _python(
      "import subprocess, sys, time; "
      "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
      f"open({str(pid_file)!r}, 'w').write(str(child.pid)); "
      "time.sleep(60)"
    )

    # when it is run with a short timeout, then it times out.
    with pytest.raises(
      common_tools.FunctionExecutionError,
      match=r"Evaluation timed-out",
    ):
      common_tools.run_command(cmd, timeout=1)

    # and the orphaned child was killed along with it.
    child_pid = int(pid_file.read_text())
    assert wait_until_dead(child_pid), (
      f"orphan {child_pid} survived the timeout and is still running on the pod"
    )

  def test_returns_when_a_timed_out_evaluator_holds_an_open_worker_pool(
    self, tmp_path: pathlib.Path
  ) -> None:
    """
    An evaluator that overruns while holding an open worker pool still fails with
    the timeout.
    """
    # this evaluator will overrun because its pool workers are still
    # working, each holding a copy of the evaluator's stdout and stderr.
    script = tmp_path / "pool_evaluator.py"
    script.write_text(
      "import concurrent.futures, time\n"
      "\n"
      "def work(i):\n"
      "  time.sleep(30)\n"
      "  return i\n"
      "\n"
      "if __name__ == '__main__':\n"
      "  pool = concurrent.futures.ProcessPoolExecutor(max_workers=2)\n"
      "  list(pool.map(work, range(2)))\n"
    )

    start = time.monotonic()
    with pytest.raises(
      common_tools.FunctionExecutionError,
      match=r"Evaluation timed-out",
    ):
      common_tools.run_command([sys.executable, str(script)], timeout=5)

    # check that the test didn't block on the pool workers' 30s sleep.
    assert time.monotonic() - start < 10

  def test_warns_when_output_cannot_be_drained(
    self, caplog: pytest.LogCaptureFixture, tmp_path: pathlib.Path
  ) -> None:
    """
    Abandoning a reader is reported at WARNING, naming the stuck stream.

    A timed-out command reports only the timeout to the caller, so the log is the
    only place its leaked descendant surfaces.
    """
    # given an evaluator whose child leaves the process group, so it survives the
    # kill and keeps holding the inherited pipes. It outlives the shortened reader
    # deadline, so the join has to give up on it. The pid goes to a file so this
    # test can clean up what the code under test cannot.
    pid_file = tmp_path / "escaper.pid"
    cmd = _python(
      "import subprocess, sys, os, time; "
      "child = subprocess.Popen("
      "  [sys.executable, '-c', 'import time; time.sleep(3)'], "
      "  preexec_fn=os.setsid); "
      f"open({str(pid_file)!r}, 'w').write(str(child.pid)); "
      "time.sleep(60)"
    )

    # when it times out.
    try:
      with (
        caplog.at_level(logging.WARNING),
        pytest.raises(common_tools.FunctionExecutionError, match=r"timed-out"),
      ):
        common_tools.run_command(cmd, timeout=1)
    finally:
      if pid_file.exists():
        with contextlib.suppress(ProcessLookupError):
          os.kill(int(pid_file.read_text()), signal.SIGKILL)

    # then the undrained output is reported, naming both stuck readers.
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("Output not drained" in w for w in warnings), warnings
    assert any("stdout" in w and "stderr" in w for w in warnings), warnings

  def test_returns_on_its_own_deadline_when_a_descendant_escapes_the_group(
    self, tmp_path: pathlib.Path
  ) -> None:
    """
    A descendant that escapes the process group cannot extend the call, and is
    recorded as contaminating the pod.
    """
    # given an evaluator whose child escapes the process group and holds the pipes
    # for far longer than any deadline in play.
    pid_file = tmp_path / "escaper.pid"
    cmd = _python(
      "import subprocess, sys, os; "
      "child = subprocess.Popen("
      "  [sys.executable, '-c', 'import time; time.sleep(30)'], "
      "  preexec_fn=os.setsid); "
      f"open({str(pid_file)!r}, 'w').write(str(child.pid)); "
      "print('fitness: 1.0')"
    )

    # when it is run.
    start = time.monotonic()
    try:
      with pytest.raises(
        common_tools.FunctionExecutionError,
        match=r"left background processes still holding its output",
      ):
        common_tools.run_command(cmd, timeout=30)
      elapsed = time.monotonic() - start
    finally:
      if pid_file.exists():
        with contextlib.suppress(ProcessLookupError):
          os.kill(int(pid_file.read_text()), signal.SIGKILL)

    # then it returned on its own deadline, nowhere near the escapee's 30s.
    assert elapsed < 10, f"took {elapsed:.1f}s, so cleanup waited on the escapee"

    # and the pod is marked unfit for further measurements.
    assert common_tools.is_contaminated()

  def test_kills_orphan_that_released_the_output_pipes(self) -> None:
    """
    A descendant that closes the output pipes lets the evaluation succeed, but is
    still cleaned up so it cannot compete with the next evaluation.
    """
    # given an evaluator whose child redirects its own output to devnull, so the
    # pipes drain normally and the result is trustworthy.
    cmd = _python(
      "import subprocess, sys; "
      "child = subprocess.Popen("
      "  [sys.executable, '-c', 'import time; time.sleep(60)'], "
      "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
      "print(child.pid)"
    )

    # when it is run, then the evaluation succeeds.
    result = common_tools.run_command(cmd, timeout=30)

    # and the surviving child was still swept up.
    child_pid = int(result.strip())
    assert wait_until_dead(child_pid), (
      f"orphan {child_pid} was left running after a successful evaluation"
    )


class TestLastOutputLines:
  """
  Tests for the `last_output_lines` helper.
  """

  def test_returns_tail_of_both_streams_with_headers(self) -> None:
    """
    The last N lines of both stdout and stderr are returned, each under its own
    underlined header.
    """
    # given stdout and stderr each with more lines than the requested maximum.
    stdout = "out1\nout2\nout3"
    stderr = "err1\nerr2\nerr3"

    # when the last two lines of each stream are requested.
    result = common_tools.last_output_lines(stdout, stderr, max_lines=2)

    # then the tail of both streams is returned under underlined headers.
    assert result == (
      "stdout (last 2 lines)\n"
      "---------------------\n"
      "out2\nout3\n\n"
      "stderr (last 2 lines)\n"
      "---------------------\n"
      "err2\nerr3"
    )

  def test_reports_no_output_placeholder_for_empty_stream(self) -> None:
    """
    A stream that produced no output is reported with an explicit placeholder
    rather than being omitted.
    """
    # given output on stdout only.
    stdout = "out1\nout2\nout3"
    stderr = "   \n  "

    # when the last two lines of each stream are requested.
    result = common_tools.last_output_lines(stdout, stderr, max_lines=2)

    # then stderr shows the no-output placeholder and stdout shows its tail.
    assert result == (
      "stdout (last 2 lines)\n"
      "---------------------\n"
      "out2\nout3\n\n"
      "stderr (last 2 lines)\n"
      "---------------------\n"
      "<No output>"
    )

  def test_reports_no_output_for_both_empty_streams(self) -> None:
    """
    Empty stdout and stderr both show the no-output placeholder under their
    headers.
    """
    # given no output at all.
    # when the last two lines of each stream are requested.
    result = common_tools.last_output_lines("", "", max_lines=2)

    # then both streams show the no-output placeholder.
    assert result == (
      "stdout (last 2 lines)\n"
      "---------------------\n"
      "<No output>\n\n"
      "stderr (last 2 lines)\n"
      "---------------------\n"
      "<No output>"
    )

  def test_returns_all_lines_when_fewer_than_max(self) -> None:
    """
    When a stream has fewer lines than the maximum, every line is returned
    unchanged.
    """
    # given streams with fewer lines than the requested maximum.
    stdout = "only out line"
    stderr = "only err line"

    # when up to ten lines of each stream are requested.
    result = common_tools.last_output_lines(stdout, stderr, max_lines=10)

    # then all available lines are returned under their headers.
    assert result == (
      "stdout (last 10 lines)\n"
      "----------------------\n"
      "only out line\n\n"
      "stderr (last 10 lines)\n"
      "----------------------\n"
      "only err line"
    )

  def test_uses_default_max_lines_in_header(self) -> None:
    """
    When `max_lines` is omitted the default is used in the header text.
    """
    # given output on both streams and no explicit maximum.
    # when the last lines are requested using the default maximum.
    result = common_tools.last_output_lines("out", "err")

    # then the header reflects the module default of twenty lines.
    assert result == (
      "stdout (last 20 lines)\n"
      "----------------------\n"
      "out\n\n"
      "stderr (last 20 lines)\n"
      "----------------------\n"
      "err"
    )

"""Tests for the common_tools module."""

import sys

import pytest

import common_tools


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
      match=r"^The evaluator produced no output\.$",
    ):
      common_tools.run_command(cmd)

  def test_raises_exit_code_and_tail_on_nonzero_exit(self) -> None:
    """
    A command that exits non-zero reports its exit code together with the tail
    of its output.
    """
    # given a command that prints two lines then exits non-zero.
    cmd = _python(
      "print('some earlier noisy output'); "
      "print('evaluator failed on its final output line'); "
      "import sys; sys.exit(2)"
    )

    # when it is run, then the error carries the exit code and the output tail.
    with pytest.raises(common_tools.FunctionExecutionError) as exc_info:
      common_tools.run_command(cmd)

    assert str(exc_info.value) == (
      "The evaluator returned a non-zero exit code (2) with the following "
      "output:\n\n"
      "some earlier noisy output\n"
      "evaluator failed on its final output line"
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
      match=r"^Execution failed: Terminated by signal 11 \(SIGSEGV\)",
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
      match=r"^Execution failed: Timeout$",
    ):
      common_tools.run_command(cmd, timeout=0.5)


class TestLastOutputLines:
  """
  Tests for the `last_output_lines` helper.
  """

  def test_combines_stdout_and_stderr_and_keeps_tail(self) -> None:
    """
    The last N lines of stdout followed by stderr are returned.
    """
    # given stdout and stderr with more lines than the requested maximum.
    stdout = "out1\nout2\nout3"
    stderr = "err1\nerr2"

    # when the last three lines are requested.
    result = common_tools.last_output_lines(stdout, stderr, max_lines=3)

    # then the tail spanning stdout into stderr is returned.
    assert result == "out3\nerr1\nerr2"

  def test_returns_empty_string_when_no_output(self) -> None:
    """
    Empty stdout and stderr yield an empty string.
    """
    # given no output at all.
    # when the last lines are requested, then the result is empty.
    assert common_tools.last_output_lines("", "") == ""

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
      "stderr (last 10 lines)\n"
      "----------------------\n"
      "evaluator failed on stderr\n\n"
      "stdout (last 10 lines)\n"
      "----------------------\n"
      "some earlier noisy output\n"
      "evaluator progress on stdout"
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
      "stderr (last 2 lines)\n"
      "---------------------\n"
      "err2\nerr3\n\n"
      "stdout (last 2 lines)\n"
      "---------------------\n"
      "out2\nout3"
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
      "stderr (last 2 lines)\n"
      "---------------------\n"
      "<No output>\n\n"
      "stdout (last 2 lines)\n"
      "---------------------\n"
      "out2\nout3"
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
      "stderr (last 2 lines)\n"
      "---------------------\n"
      "<No output>\n\n"
      "stdout (last 2 lines)\n"
      "---------------------\n"
      "<No output>"
    )

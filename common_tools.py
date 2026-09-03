"""Common functionality across sandboxex."""

import contextlib
import io
import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Generator, MutableSequence, Sequence
from typing import IO

import requests

logger = logging.getLogger(__name__)


def read_stream(
  stream: IO[str], output_list: MutableSequence[str], label: str = ""
) -> None:
  """Read a stream line-by-line, logging each line and appending it to
  ``output_list``.
  """
  try:
    for line in iter(stream.readline, ""):
      output_list.append(line)
      logger.info("[%s] %s", label, line.rstrip("\r\n"), extra={"category": "user"})
  except (io.UnsupportedOperation, UnicodeDecodeError, ValueError) as e:
    # ValueError("I/O operation on closed file"): an abandoned reader's `finally`
    # closes the stream mid-`readline`, and a stray traceback would litter the log.
    output_list.append(f"[Error reading stream] {e}")
  finally:
    stream.close()


class FunctionExecutionError(Exception):
  """Exception raised when a function execution fails."""


def error_code_to_string(sig: int) -> str:
  """Convert a signal code to a string."""
  sig_name = signal.Signals(sig).name
  sig_desc = signal.strsignal(sig)
  return f"Terminated by signal {sig} ({sig_name}): {sig_desc}"


# The number of trailing output lines reported for each stream when the
# evaluator exits with a non-zero status.
MAX_ERROR_OUTPUT_LINES = 20

# Placeholder shown for a stream that produced no output.
_NO_OUTPUT_PLACEHOLDER = "<No output>"

# How long to wait for a reader thread to see end-of-stream. A stream ends only
# once every process holding a copy of the pipe has exited, so a descendant that
# outlives the command (a worker pool, a spawned server, a backgrounded shell
# job) holds it open indefinitely.
#
# Once the command itself is gone, at most one pipe buffer (64 KiB, ~800 lines)
# remains to drain, well under a millisecond. The rest of this budget covers a
# straggling descendant still shutting down.
READER_JOIN_TIMEOUT = 5.0

# How long to give a killed process group to die. `SIGKILL` cannot be caught, so
# this covers only the kernel's own cleanup; a process wedged in an
# uninterruptible syscall can outlast it.
PROCESS_GROUP_KILL_TIMEOUT = 5.0

# Set when a command leaves a process this daemon could not kill. The survivor
# competes for CPU, memory and I/O with whatever runs next, so a candidate timed
# against it is scored on a machine that is not idle.
_contaminated = threading.Event()


def mark_contaminated() -> None:
  """Record that a process survived this daemon's attempt to kill it."""
  _contaminated.set()


def is_contaminated() -> bool:
  """Whether a command left a process this daemon could not kill."""
  return _contaminated.is_set()


def _format_stream_tail(name: str, output: str, max_lines: int) -> str:
  """Format the tail of a single output stream with an underlined header.

  Args:
    name: The stream name to show in the header (for example "stdout").
    output: The full output captured from the stream.
    max_lines: The maximum number of trailing lines to include.

  Returns:
    An underlined header naming the stream followed by its last `max_lines`
    lines, or a placeholder when the stream produced no output.
  """
  header = f"{name} (last {max_lines} lines)"
  underline = "-" * len(header)
  lines = output.strip().splitlines()[-max_lines:]
  body = "\n".join(lines) if lines else _NO_OUTPUT_PLACEHOLDER
  return f"{header}\n{underline}\n{body}"


def last_output_lines(
  stdout: str, stderr: str, max_lines: int = MAX_ERROR_OUTPUT_LINES
) -> str:
  """Return the last few lines of both evaluator output streams.

  Each stream is given an underlined header, and a stream that produced no output
  is reported with a placeholder.

  Args:
    stdout: The evaluator's standard output.
    stderr: The evaluator's standard error.
    max_lines: The maximum number of trailing lines to include per stream.

  Returns:
    The last `max_lines` lines of stdout followed by the last `max_lines` lines
    of stderr, each under its own underlined header.
  """
  return (
    f"{_format_stream_tail('stdout', stdout, max_lines)}\n\n"
    f"{_format_stream_tail('stderr', stderr, max_lines)}"
  )


def kill_process_group(process: subprocess.Popen) -> None:
  """Kill `process` and every descendant it started.

  Requires `process` to have been started with `start_new_session=True`, which
  makes it the leader of its own process group. Safe to call after `process` has
  exited: the group outlives its leader while any member runs, so this also reaps
  orphans left by a clean exit.

  Signals only. `release_process` does the bounded wait for the leader to die.

  Best effort — a descendant that calls `setsid()` or double-forks leaves the
  group and survives.
  """
  # The group id is the leader's pid. `os.getpgid()` would fail once the leader
  # has been reaped, while the rest of the group is still running.
  try:
    os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    # The group is already gone.
    pass


def start_stream_readers(
  process: subprocess.Popen,
  stdout_sink: MutableSequence[str],
  stderr_sink: MutableSequence[str],
) -> list[threading.Thread]:
  """Start background threads draining `process`'s stdout and stderr.

  Each line is logged as it arrives and appended to the matching sink. The threads
  are daemons, so a reader left blocked on a pipe cannot prevent interpreter
  shutdown.
  """
  # The caller passed stdout/stderr=PIPE, so both are open.
  assert process.stdout is not None
  assert process.stderr is not None

  # Each thread is named after its stream, which `join_stream_readers` reports.
  readers = [
    threading.Thread(
      target=read_stream,
      args=(process.stdout, stdout_sink, "stdout"),
      name="stdout",
      daemon=True,
    ),
    threading.Thread(
      target=read_stream,
      args=(process.stderr, stderr_sink, "stderr"),
      name="stderr",
      daemon=True,
    ),
  ]
  for reader in readers:
    reader.start()
  return readers


def join_stream_readers(readers: Sequence[threading.Thread]) -> bool:
  """Wait for the reader threads to finish, up to `READER_JOIN_TIMEOUT` total.

  Returns True if every reader reached end-of-stream, and False if any is still
  blocked on a pipe that something else is holding open.
  """
  deadline = time.monotonic() + READER_JOIN_TIMEOUT
  for reader in readers:
    reader.join(timeout=max(0.0, deadline - time.monotonic()))

  stuck = [reader.name for reader in readers if reader.is_alive()]
  if stuck:
    logger.warning(
      "Output not drained after %.0fs: %s still blocked, so a process is holding "
      "the pipe. Abandoning the reader(s) to release the sandbox.",
      READER_JOIN_TIMEOUT,
      ", ".join(stuck),
    )
    mark_contaminated()
    return False

  return True


def release_process(process: subprocess.Popen) -> None:
  """Reap `process`, waiting at most `PROCESS_GROUP_KILL_TIMEOUT`.

  The output pipes are left to the reader threads, which close them in
  `read_stream`'s `finally`. Closing a stream blocks while a reader thread is
  inside `readline` on it, for as long as whatever holds the pipe open.
  """
  try:
    process.wait(timeout=PROCESS_GROUP_KILL_TIMEOUT)
  except subprocess.TimeoutExpired:
    # The group was already sent SIGKILL, so the process is wedged in the kernel
    # and stays a zombie until the container restarts.
    logger.warning("Process %d could not be reaped.", process.pid)
    mark_contaminated()


@contextlib.contextmanager
def sandboxed_process(
  process: subprocess.Popen,
) -> Generator[subprocess.Popen, None, None]:
  """Kill and reap `process`'s group on the way out, however the block ends."""
  try:
    yield process
  finally:
    # A reaped leader's pid can already belong to an unrelated group.
    if process.poll() is None:
      kill_process_group(process)
    release_process(process)


def run_command(
  cmd: str | Sequence[str],
  cwd: str = ".",
  timeout: float = 10.0,
) -> str:
  """Run a command with timeout and return the last line of its output.

  The command runs in its own process group, and both the wait for it to exit and
  the wait for its output to drain are bounded, so this returns even when the
  command leaves background processes behind.
  """

  # PyInstaller can poison the subprocess environemnt. See below
  # https://pyinstaller.org/en/stable/runtime-information.html?ld-library-path-libpath-considerations=#ld-library-path-libpath-considerations
  env = os.environ.copy()
  orig = env.pop("LD_LIBRARY_PATH_ORIG", None)
  if orig:
    env["LD_LIBRARY_PATH"] = orig
  else:
    env.pop("LD_LIBRARY_PATH", None)
  # Also configure Python to use unbuffered output so we can stream the output 
  env["PYTHONUNBUFFERED"] = "1"

  process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=cwd,
    bufsize=1,  # Allows real-time output
    universal_newlines=True,
    text=True,
    env=env,
    # Own process group, so everything the command spawns can be cleaned up
    # together. See `kill_process_group`.
    start_new_session=True,
  )
  with sandboxed_process(process):
    # `read_stream` logs every line live, so only the last
    # `MAX_ERROR_OUTPUT_LINES` of each stream are retained here: enough for the
    # error tail and the final stdout line, and bounded no matter how much the
    # evaluator prints.
    stdout_lines: deque[str] = deque(maxlen=MAX_ERROR_OUTPUT_LINES)
    stderr_lines: deque[str] = deque(maxlen=MAX_ERROR_OUTPUT_LINES)
    readers = start_stream_readers(process, stdout_lines, stderr_lines)

    try:
      returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
      kill_process_group(process)
      join_stream_readers(readers)
      raise FunctionExecutionError(
        f"Evaluation timed-out after {timeout} seconds."
      ) from exc

    # Drain before killing the group: killing closes the pipes, and a pipe still
    # open is the evidence that a descendant outlived the evaluator.
    drained = join_stream_readers(readers)
    if not drained:
      kill_process_group(process)
      raise FunctionExecutionError(
        "The evaluator exited but left background processes still holding its "
        "output, so its result could not be collected. Ensure the evaluator "
        "terminates every process it starts before exiting."
      )

    # The output drained, so the result below is trustworthy. Descendants that
    # closed or never inherited the pipes can still be running, and would compete
    # with the next evaluation for this pod's CPU and memory.
    kill_process_group(process)

    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)

    if returncode < 0:
      # The process was killed by a signal (for example, SIGSEGV).
      raise FunctionExecutionError(error_code_to_string(-returncode))
    if returncode != 0:
      # Ran to completion but exited non-zero. Report the exit code with the tail
      # of its output.
      message = last_output_lines(stdout, stderr)
      raise FunctionExecutionError(
        f"The evaluator returned a non-zero exit code "
        f"({returncode}) with the following output:\n\n{message}"
      )
    lines = stdout.strip().splitlines()
    if not lines:
      # Exited cleanly but printed nothing, so there is no result to parse.
      raise FunctionExecutionError("Evaluator Format Error: No output was written.")
    return lines[-1]  # Return only the last line of output


def wait_for_url(url: str, timeout: int = 300, interval: int = 1) -> bool:
  """
  Keep checking a URL until it returns a response or timeout is reached.

  Args:
      url (str): The URL to check.
      timeout (int): Total time to keep trying (in seconds).
      interval (int): Time to wait between retries (in seconds).

  Returns:
      bool: True if `url` is available.
  """
  start_time = time.time()

  while time.time() - start_time < timeout:
    try:
      response = requests.get(url)
      if response.status_code == 200:
        return True
    except requests.RequestException:
      # Optionally log the exception or just continue
      pass
    print(f"Waiting for {url} to be available...")
    time.sleep(interval)

  return False


def stop_and_remove_image(image_name: str) -> None:
  """Stop and remove a Docker image."""

  # Step 1: Find running container for the image
  containers = (
    subprocess.check_output(
      ["docker", "ps", "-q", "--filter", f"ancestor={image_name}"]
    )
    .decode()
    .strip()
    .splitlines()
  )

  for container_id in containers:
    # Step 2: Stop the container
    subprocess.run(
      ["docker", "stop", container_id],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      check=False,
    )
    # Step 3: Remove the container
    subprocess.run(
      ["docker", "rm", container_id],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      check=False,
    )

  # Step 4: Remove the image in the background
  subprocess.Popen(
    ["docker", "rmi", image_name],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
  )

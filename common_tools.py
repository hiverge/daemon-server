"""Common functionality across sandboxex."""

import io
import signal
import subprocess
import time

import requests

def read_stream(stream, output_list):
  """Helper function to read stream line by line and store it in a list."""
  try:
    for line in iter(stream.readline, ""):
      output_list.append(line)
  except (io.UnsupportedOperation, UnicodeDecodeError) as e:
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
# evaluator exits with a non-zero status. Showing the tail (rather than
# everything) keeps the error message focused on where the failure surfaced
# while still giving context.
MAX_ERROR_OUTPUT_LINES = 10

# Placeholder shown for a stream that produced no output.
_NO_OUTPUT_PLACEHOLDER = "<No output>"


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

  Both streams are shown because each can carry useful context: stderr usually
  holds the failure itself, while stdout can reveal how far the evaluator got
  before it crashed. Each stream is given an underlined header, and a stream
  that produced no output is reported explicitly rather than omitted.

  Args:
    stdout: The evaluator's standard output.
    stderr: The evaluator's standard error.
    max_lines: The maximum number of trailing lines to include per stream.

  Returns:
    The last `max_lines` lines of stderr followed by the last `max_lines` lines
    of stdout, each under its own underlined header.
  """
  return (
    f"{_format_stream_tail('stderr', stderr, max_lines)}\n\n"
    f"{_format_stream_tail('stdout', stdout, max_lines)}"
  )


def run_command(
  cmd: str,
  cwd: str = ".",
  timeout: float = 10.0,
) -> str:
  """Run a command with timeout and return the output."""

  with subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=cwd,
    bufsize=1,  # Allows real-time output
    universal_newlines=True,
    text=True,
  ) as process:
    try:
      stdout, stderr = process.communicate(timeout=timeout)
      if process.returncode < 0:
        # The process was killed by a signal (for example, SIGSEGV).
        raise FunctionExecutionError(error_code_to_string(-process.returncode))
      if process.returncode != 0:
        # The evaluator ran to completion but exited non-zero, so the evaluation
        # did not complete cleanly. Report the exit code with the tail of its
        # output so the user can see where it failed.
        message = last_output_lines(stdout, stderr)
        raise FunctionExecutionError(
          f"The evaluator returned a non-zero exit code "
          f"({process.returncode}) with the following output:\n\n{message}"
        )
      lines = stdout.strip().splitlines()
      if not lines:
        # Exited cleanly but printed nothing, so there is no result to parse.
        raise FunctionExecutionError("Evaluator Format Error: No output was written.")
      return lines[-1]  # Return only the last line of output
    except subprocess.TimeoutExpired as exc:
      process.kill()
      raise FunctionExecutionError(f"Evaluation timed-out after {timeout} seconds.") from exc


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


def stop_and_remove_image(image_name: str):
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

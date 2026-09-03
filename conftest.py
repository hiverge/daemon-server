"""Helpers shared by the test modules."""

import os
import time


def wait_until_dead(pid: int, timeout: float = 5.0) -> bool:
  """
  Poll until `pid` disappears, returning whether it did within `timeout`.

  `SIGKILL` delivery to a whole process group is not instantaneous.
  """
  deadline = time.monotonic() + timeout
  while True:
    try:
      os.kill(pid, 0)
    except ProcessLookupError:
      return True
    if time.monotonic() >= deadline:
      return False
    time.sleep(0.05)

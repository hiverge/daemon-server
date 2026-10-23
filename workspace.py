"""Linux session filesystems and synchronous, isolated workspace commands."""

from __future__ import annotations

import ctypes
import json
import math
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

import common_tools


class WorkspaceError(Exception):
  def __init__(self, message: str, status_code: int = 503):
    super().__init__(message)
    self.status_code = status_code


# Every persistent writable mount has one directory under the session root.
# /var/tmp is an alias of /tmp; /dev and /proc are recreated for each command.
WRITABLE_MOUNTS = {
  "merged": "/app",
  "home": "/home/agent",
  "tmp": "/tmp",
  "run": "/run",
  "shm": "/dev/shm",
}
TOOL_PATHS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt")
CONFIG_PATHS = (
  "/etc/passwd",
  "/etc/group",
  "/etc/nsswitch.conf",
  "/etc/hosts",
  "/etc/localtime",
  "/etc/ssl",
)
COMMAND_ENV = {
  "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
  "HOME": "/home/agent",
  "TMPDIR": "/tmp",
  "LANG": "C.UTF-8",
  "PYTHONUNBUFFERED": "1",
}
CLEANUP_TIMEOUT = 5.0
SETUP_TIMEOUT = 30.0


class WorkspaceRuntime:
  """One assignment at a time, serialized by the daemon's sandbox_lock.

  The caller prepares an immutable baseline and a private, disk-backed storage
  directory. A failed cleanup permanently retires this instance.
  """

  def __init__(self, baseline: str | Path, storage: str | Path):
    self.baseline = Path(baseline).resolve()
    self.storage = Path(storage).resolve()
    self.session_dir: Path | None = None
    self.assignment_token: str | None = None
    self.used_tokens: set[str] = set()
    self.retired = False
    self.ready = False

  @property
  def state(self) -> str:
    if self.retired:
      return "retired"
    if not self.ready:
      return "starting"
    return "active" if self.assignment_token else "idle"

  def initialize(self) -> None:
    """Exercise the actual mount/namespace configuration before serving."""
    try:
      if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("workspace mode requires Linux and uid 0")
      if (
        not hasattr(os, "pidfd_open")
        or not hasattr(os, "P_PIDFD")
        or not hasattr(signal, "pidfd_send_signal")
      ):
        raise RuntimeError("workspace mode requires Linux pidfd support")
      for helper in ("mount", "umount", "bwrap", "bash"):
        if shutil.which(helper) is None:
          raise RuntimeError(f"workspace mode requires the {helper} helper")
      # Bubblewrap's monitor exits when the shell does. Adopt its namespace init
      # so repeated commands do not leave zombies with an unrelated container
      # PID 1 (including the frozen binary's bootloader).
      libc = ctypes.CDLL(None, use_errno=True)
      if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot enable workspace child subreaper")
      if not self.baseline.is_dir():
        raise RuntimeError("workspace baseline must be an existing directory")
      if self.storage.is_relative_to(self.baseline) or self.baseline.is_relative_to(
        self.storage
      ):
        raise RuntimeError("workspace baseline and storage must be disjoint")
      for path in (*TOOL_PATHS, *CONFIG_PATHS):
        root = Path(path).resolve()
        if self.storage.is_relative_to(root) or self.baseline.is_relative_to(root):
          raise RuntimeError(
            "workspace baseline/storage cannot be inside the tool environment"
          )
      # These characters have syntax in OverlayFS's mount option parser.
      if any(c in str(self.baseline) + str(self.storage) for c in ",:\\\n"):
        raise RuntimeError(
          "workspace paths cannot contain comma, colon, backslash or newline"
        )
      self.storage.mkdir(parents=True, exist_ok=True, mode=0o700)
      if any(self.storage.iterdir()):
        raise RuntimeError("workspace storage is not empty; retire this container")
      # Reject a memory-backed upper/work volume, including an enclosing tmpfs.
      mounts = Path("/proc/self/mountinfo").read_text().splitlines()
      enclosing = [
        line
        for line in mounts
        if self.storage.is_relative_to(self._mount_path(line.split()[4]))
      ]
      backing = max(enclosing, key=lambda line: len(line.split()[4]))
      if any(
        field.startswith("shared:") for field in backing.split(" - ", 1)[0].split()[6:]
      ):
        raise RuntimeError(
          "workspace storage requires private mount propagation; launch with unshare --mount --propagation private"
        )
      if backing.split(" - ", 1)[1].split()[0] in ("tmpfs", "ramfs"):
        raise RuntimeError("workspace storage must be disk-backed, not tmpfs/ramfs")
      self._create_filesystem()
      output, returncode = self._execute("true", "/app", SETUP_TIMEOUT)
      if returncode != 0:
        raise RuntimeError(f"workspace namespace probe failed: {output}")
      self._destroy_filesystem()
      self.ready = True
    except BaseException as exc:
      self.retired = True
      # Also attempt to unmount a partially created/probed filesystem.
      self._destroy_filesystem()
      if isinstance(exc, Exception):
        raise WorkspaceError(
          f"Workspace startup failed (check OverlayFS, namespaces and mount permissions): {exc}"
        ) from exc
      raise

  @staticmethod
  def _mount_path(value: str) -> Path:
    for escaped, char in (
      ("\\040", " "),
      ("\\011", "\t"),
      ("\\012", "\n"),
      ("\\134", "\\"),
    ):
      value = value.replace(escaped, char)
    return Path(value)

  def _check_available(self) -> None:
    if self.retired or not self.ready:
      raise WorkspaceError("Workspace unavailable; retire this runtime")

  @staticmethod
  def validate_token(token: str) -> None:
    if not isinstance(token, str) or not token or len(token) > 256:
      raise WorkspaceError(
        "assignment_token must be a nonempty string of at most 256 characters", 400
      )

  def check_assignment(self, token: str) -> None:
    self.validate_token(token)
    self._check_available()
    if self.assignment_token is None or token != self.assignment_token:
      raise WorkspaceError("Stale or unknown workspace assignment", 409)

  def start(self, token: str) -> None:
    self.validate_token(token)
    self._check_available()
    if self.assignment_token is not None:
      raise WorkspaceError("Workspace already assigned", 409)
    if token in self.used_tokens:
      raise WorkspaceError("Assignment token has already been used", 409)
    self.used_tokens.add(token)
    try:
      self._create_filesystem()
    except BaseException:
      self.retired = True
      self._destroy_filesystem()
      raise WorkspaceError("Workspace filesystem creation failed; retire this runtime")
    self.assignment_token = token

  def release(self, token: str) -> None:
    self.check_assignment(token)
    # Each command has already quiesced its PID namespace while holding the
    # same request lock. Only a confirmed unmount/removal permits reassignment.
    self._destroy_filesystem()
    self.assignment_token = None

  def _admin_command(self, args: list[str]) -> None:
    process = subprocess.Popen(
      args,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      start_new_session=True,
      env={"PATH": COMMAND_ENV["PATH"]},
    )
    stdout: list[str] = []
    stderr: list[str] = []
    with common_tools.sandboxed_process(process):
      readers = common_tools.start_stream_readers(process, stdout, stderr)
      try:
        returncode = process.wait(timeout=SETUP_TIMEOUT)
      finally:
        if process.poll() is None:
          common_tools.kill_process_group(process)
        drained = common_tools.join_stream_readers(readers)
    if not drained or common_tools.is_contaminated():
      raise RuntimeError(f"{args[0]} cleanup failed")
    if returncode:
      raise RuntimeError(f"{args[0]} failed: {''.join(stderr).strip()}")

  def _create_filesystem(self) -> None:
    self.session_dir = Path(tempfile.mkdtemp(prefix="session-", dir=self.storage))
    for name in ("upper", "work", *WRITABLE_MOUNTS):
      (self.session_dir / name).mkdir()
    options = (
      f"lowerdir={self.baseline},upperdir={self.session_dir / 'upper'},"
      f"workdir={self.session_dir / 'work'},index=off"
    )
    self._admin_command(
      [
        "mount",
        "--internal-only",
        "-t",
        "overlay",
        "overlay",
        "-o",
        options,
        str(self.session_dir / "merged"),
      ]
    )

  def _destroy_filesystem(self) -> None:
    if self.session_dir is None:
      return
    try:
      merged = self.session_dir / "merged"
      if os.path.ismount(merged):
        # Never lazy-unmount: that would acknowledge reset with live users.
        self._admin_command(["umount", "--internal-only", str(merged)])
      if os.path.ismount(merged):
        raise RuntimeError("workspace mount is still active")
      shutil.rmtree(self.session_dir)
      self.session_dir = None
    except BaseException as exc:
      self.retired = True
      raise WorkspaceError("Workspace reset failed; retire this runtime") from exc

  def _command(self, cmd: str, cwd: str, info_fd: int, block_fd: int) -> list[str]:
    assert self.session_dir is not None
    args = [
      "bwrap",
      "--unshare-user",
      "--disable-userns",
      "--unshare-pid",
      "--unshare-net",
      "--unshare-ipc",
      "--unshare-uts",
      "--die-with-parent",
      "--new-session",
      "--cap-drop",
      "ALL",
      "--info-fd",
      str(info_fd),
      "--block-fd",
      str(block_fd),
      "--clearenv",
    ]
    # Build a small root rather than exposing the daemon's root, credentials,
    # baseline, OverlayFS metadata, or other sessions through a recursive bind.
    for path in TOOL_PATHS:
      if Path(path).is_symlink():
        args += ["--symlink", os.readlink(path), path]
      elif Path(path).exists():
        args += ["--ro-bind", path, path]
    for path in CONFIG_PATHS:
      if Path(path).exists():
        args += ["--ro-bind", path, path]
    args += ["--proc", "/proc", "--remount-ro", "/proc", "--dev", "/dev"]
    for name, destination in WRITABLE_MOUNTS.items():
      args += ["--bind", str(self.session_dir / name), destination]
    args += [
      "--dir",
      "/var",
      "--symlink",
      "/tmp",
      "/var/tmp",
      "--symlink",
      "/run",
      "/var/run",
      "--remount-ro",
      "/dev",
      "--remount-ro",
      "/",
      "--chdir",
      cwd,
    ]
    for name, value in COMMAND_ENV.items():
      args += ["--setenv", name, value]
    return args + ["--", "/bin/bash", "--noprofile", "--norc", "-c", cmd]

  def shell(self, token: str, cmd: str, cwd: str, timeout: float) -> str:
    self.check_assignment(token)
    if not isinstance(cmd, str) or "\0" in cmd:
      raise WorkspaceError("cmd must be a string without NUL characters", 400)
    if not isinstance(cwd, str) or "\0" in cwd:
      raise WorkspaceError("cwd must be a string without NUL characters", 400)
    if not math.isfinite(timeout) or timeout <= 0:
      raise WorkspaceError("timeout must be positive and finite", 400)
    # Resolve inside the command's root, not against the daemon's host view.
    cwd = str(PurePosixPath("/app") / cwd)
    try:
      output, _ = self._execute(cmd, cwd, timeout)
    except WorkspaceError:
      raise
    except Exception as exc:
      raise WorkspaceError("Workspace command could not be executed") from exc
    return output

  def _open_namespace(self, info_fd: int, deadline: float) -> int:
    """Pin namespace init before allowing any user command to execute.

    A pidfd avoids PID reuse races. Bubblewrap's monitor can exit as soon as the
    foreground shell does, while its namespace init still has detached children.
    """
    data = b""
    while True:
      remaining = deadline - time.monotonic()
      if remaining <= 0 or not select.select([info_fd], [], [], remaining)[0]:
        raise RuntimeError("timed out creating command namespaces")
      chunk = os.read(info_fd, 4096)
      if not chunk:
        raise RuntimeError("Bubblewrap failed before creating command namespaces")
      data += chunk
      try:
        info = json.loads(data)
      except json.JSONDecodeError:
        continue
      return os.pidfd_open(info["child-pid"])

  def _stop_namespace(self, namespace_fd: int) -> None:
    try:
      signal.pidfd_send_signal(namespace_fd, signal.SIGKILL)
    except ProcessLookupError:
      pass
    if not select.select([namespace_fd], [], [], CLEANUP_TIMEOUT)[0]:
      raise RuntimeError("command PID namespace did not exit")

  def _execute(self, cmd: str, cwd: str, timeout: float) -> tuple[str, int | None]:
    info_read, info_write = os.pipe()
    block_read, block_write = os.pipe()
    process = None
    namespace_fd = None
    readers = []
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    returncode = None
    timed_out = False
    deadline = time.monotonic() + timeout
    try:
      process = subprocess.Popen(
        self._command(cmd, cwd, info_write, block_read),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
        pass_fds=(info_write, block_read),
        env=COMMAND_ENV,
      )
      os.close(info_write)
      info_write = -1
      os.close(block_read)
      block_read = -1
      readers = common_tools.start_stream_readers(
        process,
        stdout_lines,
        stderr_lines,
        log_output=False,
      )
      namespace_fd = self._open_namespace(info_read, deadline)
      os.write(block_write, b"1")
      try:
        returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
      except subprocess.TimeoutExpired:
        timed_out = True
    finally:
      # Includes Python cancellation/exceptions, not only normal exits/timeouts.
      # Keep the request lock until init has exited and its descendants are dead.
      cleanup_failed = False
      try:
        if namespace_fd is not None:
          self._stop_namespace(namespace_fd)
        elif process is not None:
          cleanup_failed = True  # No namespace identity: reuse is uncertain.
      except BaseException:
        cleanup_failed = True
      finally:
        try:
          if process is not None:
            if process.poll() is None:
              common_tools.kill_process_group(process)
            common_tools.release_process(process)
          if namespace_fd is not None and not cleanup_failed:
            try:
              # The pidfd is already readable: this cannot block on live work.
              # Reap only this namespace init, never another request's child.
              os.waitid(os.P_PIDFD, namespace_fd, os.WEXITED)
            except ChildProcessError:
              pass  # Bubblewrap already reaped init before its monitor exited.
          drained = common_tools.join_stream_readers(readers)
        except BaseException:
          self.retired = True
          raise WorkspaceError("Workspace process cleanup failed; retire this runtime")
        finally:
          for fd in (info_read, info_write, block_read, block_write, namespace_fd):
            if fd is not None and fd >= 0:
              os.close(fd)
      if cleanup_failed or not drained or common_tools.is_contaminated():
        self.retired = True
        raise WorkspaceError("Workspace command cleanup failed; retire this runtime")
    return common_tools.format_shell_output(
      stdout_lines,
      stderr_lines,
      returncode,
      timed_out,
      drained,
      timeout,
    ), returncode

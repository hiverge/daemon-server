"""Persistent repository, terminal, and file checkpoints for workspace mode.

The existing server in main owns HTTP responses and serializes command/filesystem
requests with its sandbox lock. This module owns one session's state. File
restoration happens in place; full session reset requires container replacement.
"""

import base64
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tempfile
import time

import pexpect
from pexpect import replwrap
import psutil

import common_tools


class Workspace:
  """Own the filesystem and Bash terminal for one exclusive session assignment.

  A workspace starts unassigned, accepts one session, and is permanently retired
  on release or terminal failure. It never switches sessions in place: replacing
  the container clears processes and writable state outside the repository too.
  Callers serialize command and filesystem operations; this class has no lock.
  """

  def __init__(self, repo_dir, backup_dir, resolve_path, apply_files):
    """Seed the repository from the initialized backup, without starting Bash.

    resolve_path and apply_files are the evaluation daemon's existing helpers,
    shared here to retain its path containment and text/deletion overlay format.
    """
    self.repo = Path(repo_dir).resolve()
    self.backup = Path(backup_dir).resolve()
    self.resolve_path = resolve_path
    self.apply_files = apply_files
    self.session = None
    self.terminal = None
    self.retired = False
    self.read_only = False
    self.editable = None
    self.protected = set()
    self.checkpoints = []
    self.storage = tempfile.TemporaryDirectory(prefix="workspace-")
    self.copy_tree(self.backup, self.repo)
    subprocess.run(["git", "init", str(self.repo)], check=True, capture_output=True)

  @staticmethod
  def copy_tree(source, destination):
    """Mirror repository files, including binaries/symlinks, but preserve Git.

    Checksums ensure restoration catches edits with unchanged size and mtime.
    Destination files absent from the source are removed, except Git metadata.
    """
    Path(destination).mkdir(parents=True, exist_ok=True)
    subprocess.run(
      [
        "rsync",
        "-a",
        "--checksum",
        "--delete",
        "--exclude=.git",
        f"{source}/",
        f"{destination}/",
      ],
      check=True,
      capture_output=True,
    )

  def health(self):
    """Report availability/assignment, retiring the workspace if Bash has died."""
    if self.terminal is not None and not self.terminal.child.isalive():
      self.retire()
    return {
      "status": "unavailable" if self.retired else "healthy",
      "assigned": self.session is not None,
    }

  def require_session(self, session):
    """Reject retired workspaces and requests for an unassigned or stale session."""
    self.health()
    if self.retired:
      raise RuntimeError("Workspace requires a replacement container")
    if self.session is None or session != self.session:
      raise PermissionError("Workspace is not assigned to this session")

  def retire(self):
    """Reject further work and request container exit after the HTTP response.

    This marks the workspace for replacement; it does not reset files or stop
    every process itself. main's existing restart hook performs the exit.
    """
    self.retired = True
    # main's existing after_request hook exits PID 1 after flushing the reply.
    # Kubernetes must recreate the container, with no writable session volumes.
    common_tools.mark_contaminated()

  def path(self, name, *, deletion=False):
    """Resolve a contained API path while excluding Git metadata.

    Writes/reads follow the final symlink; deletion addresses the link itself,
    matching the evaluation daemon's resolver.
    """
    if ".git" in Path(name).parts:
      raise ValueError("Git metadata is not part of the workspace API")
    path = self.resolve_path(name, follow_final_symlink=not deletion)
    if ".git" in path.relative_to(self.repo).parts:
      raise ValueError("Git metadata is not part of the workspace API")
    return path

  def writable(self, name):
    """Check session edit policy for a repository-relative path.

    An absent editable-file allowlist permits ordinary files; an empty one
    permits none. Protected files and Hive resources remain excluded.
    """
    return (
      not self.read_only
      and name not in self.protected
      and not (name == ".hive/resources" or name.startswith(".hive/resources/"))
      and (self.editable is None or name in self.editable)
    )

  def apply(self, files):
    """Validate paths and write policy, then apply a text/null deletion overlay.

    Both the supplied path and its resolved target must be writable, so aliases
    cannot bypass a protected target. Validation precedes any filesystem writes.
    Existing write targets must be regular files; null can still delete a FIFO.
    """
    if not isinstance(files, dict):
      raise ValueError("files must map paths to text or null")
    # Check the whole overlay before writing any of it.
    for name, content in files.items():
      if not isinstance(name, str) or not (content is None or isinstance(content, str)):
        raise ValueError("files must map paths to text or null")
      path = self.path(name, deletion=content is None)
      if not self.writable(name) or not self.writable(str(path.relative_to(self.repo))):
        raise PermissionError(f"Not allowed to edit {name}")
      if content is not None and path.exists() and not path.is_file():
        raise ValueError(f"Not a regular file: {name}")
    self.apply_files(files, create_parents=True)

  @staticmethod
  def entries(root):
    """Yield relative file/symlink paths, skipping Git and symlink traversal."""
    for directory, dirs, files in os.walk(root):
      dirs[:] = [name for name in dirs if name != ".git"]
      for name in files + [
        name for name in dirs if (Path(directory) / name).is_symlink()
      ]:
        if name != ".git":
          yield str((Path(directory) / name).relative_to(root))

  def snapshot(self):
    """Return the full readable UTF-8 candidate mapping, not a sparse overlay.

    Git metadata, Hive resources, non-regular/binary files, and escaping symlinks
    are omitted. The coordinator compares this mapping to its baseline to derive
    deletions and the sparse overlay used for evaluation.
    """
    files = {}
    for name in self.entries(self.repo):
      if name.startswith(".hive/resources/"):
        continue
      try:
        path = self.path(name)
        # A FIFO can block on open; devices need not produce finite file content.
        if path.is_file():
          files[name] = path.read_text(encoding="utf-8")
      except (OSError, UnicodeDecodeError, ValueError):
        # Match evaluation writeback: only contained UTF-8 files are candidates.
        continue
    return files

  def checkpoint(self):
    """Copy repository files to local checkpoint storage and return their index.

    Unlike a candidate snapshot, this preserves binaries and symlinks. Git and
    terminal state are excluded, so restoring files does not rewind the shell.
    """
    destination = Path(self.storage.name) / str(len(self.checkpoints))
    self.copy_tree(self.repo, destination)
    self.checkpoints.append(destination)
    return len(self.checkpoints) - 1

  def _restore_disallowed(self, before):
    """Discard disallowed repository edits after a shell command has finished.

    before is the file copy taken immediately before execution. Read-only
    sessions restore it wholesale; other sessions restore only restricted paths.
    This enforces resulting file state, not what a command may do while running.
    """
    if self.read_only:
      self.copy_tree(before, self.repo)
    else:
      # Preserve binary files and symlinks too, even though candidate transfers
      # use the existing text-only overlay format.
      for name in sorted(set(self.entries(before)) | set(self.entries(self.repo))):
        if self.writable(name):
          continue
        target = self.path(name, deletion=True)
        source = before / name
        if target.is_dir() and not target.is_symlink():
          shutil.rmtree(target)
        else:
          target.unlink(missing_ok=True)
        if source.exists() or source.is_symlink():
          target.parent.mkdir(parents=True, exist_ok=True)
          shutil.copy2(source, target, follow_symlinks=False)

  def _stop_background_processes(self):
    """Kill command descendants, including detached children adopted by PID 1.

    The workspace daemon is PID 1 in its dedicated container. Scan its entire
    process tree, keeping only the persistent Bash terminal. Repeat after
    killing/reaping to catch children forked during cleanup, within the existing
    kill timeout. Failure retires the workspace before files become available.
    """
    daemon = psutil.Process()
    deadline = time.monotonic() + common_tools.PROCESS_GROUP_KILL_TIMEOUT
    while True:
      children = [
        child
        for child in daemon.children(recursive=True)
        if child.pid != self.terminal.child.pid
      ]
      if not children:
        return
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise RuntimeError("Workspace command left running processes")
      for child in children:
        try:
          child.kill()
        except psutil.NoSuchProcess:
          pass
      _, alive = psutil.wait_procs(children, timeout=remaining)
      if alive:
        raise RuntimeError("Workspace command left running processes")

  def start(self, data, session):
    """Apply the initial overlay, start Bash, and bind its first checkpoint.

    data carries files and optional editable/protected/read-only policy. The
    session ID detects stale requests; it is not an authentication credential.
    An assigned or retired workspace cannot accept another session.
    """
    if not session:
      raise ValueError("Missing X-Workspace-Session-ID")
    if self.session is not None or self.retired:
      raise RuntimeError("Workspace already assigned or awaiting replacement")
    self.editable = data.get("editable_files")
    if self.editable is not None:
      self.editable = {
        str(self.path(name).relative_to(self.repo)) for name in self.editable
      }
    self.protected = {
      str(self.path(name).relative_to(self.repo))
      for name in data.get("protected_files", [])
    }
    self.apply(data.get("files", {}))
    # A read-only session must still begin from its sampled candidate.
    self.read_only = bool(data.get("read_only", False))
    try:
      self.terminal = replwrap.bash()
      self.terminal.run_command(f"builtin cd -- {shlex.quote(str(self.repo))}")
    except (pexpect.TIMEOUT, pexpect.EOF):
      self.retire()
      raise RuntimeError("Workspace terminal failed to start") from None
    self.session = session
    return {"checkpoint": self.checkpoint()}

  def run_shell(self, data, session):
    """Run cmd synchronously in the assigned Bash session and return output.

    An optional cwd changes the persistent working directory; omitting it keeps
    the current one. Output combines stdout/stderr and appends nonzero exit
    status in the existing shell format. timeout defaults to 120 seconds.

    Repository edits are checked against session policy after execution.
    Timeouts interrupt the command with Ctrl+C and retain the session if Bash
    returns to its prompt. Terminal recovery or cleanup failures retire it.
    """
    self.require_session(session)
    timeout = float(data.get("timeout", 120))
    if not math.isfinite(timeout) or timeout <= 0:
      raise ValueError("timeout must be positive and finite")
    if not isinstance(data["cmd"], str):
      raise ValueError("cmd must be a string")
    if data.get("code_files"):
      self.apply(data["code_files"])
    command = data["cmd"]
    if data.get("cwd") is not None:
      cwd = self.path(data["cwd"])
      if not cwd.is_dir():
        raise ValueError("cwd must be an existing repository directory")
      command = f"builtin cd -- {shlex.quote(str(cwd))} && {{\n{command}\n}}"
    # Sourcing a file avoids terminal line limits and supports multiline shell
    # syntax while retaining cwd, variables, functions, and aliases.
    with tempfile.TemporaryDirectory(prefix="workspace-command-") as temporary:
      temporary = Path(temporary)
      before = temporary / "before"
      script = temporary / "command.sh"
      self.copy_tree(self.repo, before)
      script.write_text(command, encoding="utf-8")
      timed_out = False
      status = 0
      try:
        try:
          output = self.terminal.run_command(
            f"builtin source {shlex.quote(str(script))}", timeout=timeout
          )
        except pexpect.TIMEOUT:
          # Ctrl+C signals the foreground process group. Also interrupt Bash
          # when it is waiting on a separate job, so it abandons the remaining
          # sourced lines. Signal each group once to avoid a second interrupt
          # arriving at the recovered prompt and disturbing the next command.
          foreground = os.tcgetpgrp(self.terminal.child.child_fd)
          if foreground != os.getpgid(self.terminal.child.pid):
            self.terminal.child.kill(signal.SIGINT)
          try:
            os.killpg(foreground, signal.SIGINT)
          except ProcessLookupError:
            pass  # The foreground job finished between lookup and interrupt.
          # Pexpect retains pre-timeout output until this expect completes.
          self.terminal.child.expect_exact(self.terminal.prompt, timeout=5)
          output = self.terminal.child.before
          timed_out = True
        else:
          # Put the status on its own line: activation scripts can decorate the
          # prompt, and that decoration is included in run_command's output.
          status_output = self.terminal.run_command(
            "builtin printf '%s\\n' \"$?\"", timeout=5
          )
          status = int(status_output.partition("\n")[0])
        output = output.replace("\r\n", "\n")
        # No surviving writer may race with policy restoration or later reads.
        self._stop_background_processes()
        self._restore_disallowed(before)
      except (
        pexpect.TIMEOUT,
        pexpect.EOF,
        RuntimeError,
        ValueError,
        OSError,
        psutil.Error,
      ) as e:
        self.retire()
        raise RuntimeError("Workspace terminal failed; end this session") from e
    if timed_out:
      output += f"\nError: command timed out after {timeout:g}s"
    elif status:
      output += f"\nExit code: {status}"
    return output or "(no output)"

  def handle(self, operation, data, session):
    """Dispatch a workspace API operation and return its result payload.

    main holds the shared request lock and wraps this result in the HTTP
    envelope. Only start accepts an unassigned workspace. Restore changes
    repository files; reset retires the entire session for container replacement.
    """
    if not isinstance(data, dict):
      raise ValueError("Request body must be an object")
    if operation == "start":
      return self.start(data, session)
    self.require_session(session)
    if operation == "read":
      path = self.path(data["path"])
      if not path.is_file():
        raise ValueError(f"Not a regular file: {data['path']}")
      if data.get("binary"):
        return {"base64": base64.b64encode(path.read_bytes()).decode("ascii")}
      return {"content": path.read_text(encoding="utf-8")}
    if operation == "list":
      return {
        "entries": sorted(
          p.name for p in self.path(data.get("path", ".")).iterdir() if p.name != ".git"
        )
      }
    if operation == "apply":
      self.apply(data["files"])
      return {}
    if operation == "snapshot":
      return {"files": self.snapshot()}
    if operation == "checkpoint":
      return {"checkpoint": self.checkpoint()}
    if operation == "restore":
      index = data["checkpoint"]
      if not isinstance(index, int) or not 0 <= index < len(self.checkpoints):
        raise ValueError("Invalid checkpoint")
      self.copy_tree(self.checkpoints[index], self.repo)
      return {}
    if operation == "reset":
      self.retire()
      return {"restart_required": True}
    raise ValueError(f"Unknown workspace operation: {operation}")

  def close(self):
    """Close the terminal and remove local checkpoint storage during tests.

    This is local resource cleanup, not a complete session reset. Production
    releases use retire and container replacement to remove all writable state.
    """
    if self.terminal is not None:
      self.terminal.child.close(force=True)
    self.storage.cleanup()

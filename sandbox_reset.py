"""Best-effort reset of a disposable Linux execution sandbox.

One controller lives across requests. At startup, prepare() inventories paths
outside the caller's exclusions, archives file contents on disk, and registers
Linux inotify directory watches. Inotify queues kernel events for changes such
as file creation, writes, renames and deletion; those events identify paths to
check, while the archive supplies the contents to restore.

For each request, begin() protects existing process identities. The caller
attempts to stop execution processes before collecting final results. finish()
checks processes again and repairs changed filesystem paths. New directory
trees are moved into quarantine; maintain() deletes them and replenishes
prepared replacement files between requests, under the same execution lock.

Startup traversal and archive storage scale with image size. Reset uses
notifications to select paths. Lost notification coverage triggers full
verification. Cleanup time and work counts are recorded for logging.

Setup services keep running. Existing process memory and GPU state are outside
the saved filesystem baseline. Privileged commands can alter daemon state and
private archives.
"""

from __future__ import annotations

import ctypes
import errno
import heapq
import io
import os
import signal
import stat
import struct
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


class ResetIncomplete(RuntimeError):
  """Cleanup failed; the sandbox requires replacement."""


class _Work:
  """Count cleanup operations for logging."""

  def __init__(self):
    self.operations = 0

  def record(self):
    self.operations += 1


def _signature(info):
  """Fingerprint an inode's current identity and detectable modifications.

  ctime helps detect writes that preserve size and mtime. Excluding atime keeps
  reads from marking an otherwise unchanged file as modified.
  """
  return (
    info.st_dev,
    info.st_ino,
    info.st_size,
    info.st_mode,
    info.st_uid,
    info.st_gid,
    info.st_mtime_ns,
    info.st_ctime_ns,
  )


def _copy_chunk_bytes(fd: int) -> int:
  """Use the filesystem's reported preferred IO size."""
  size = os.fstat(fd).st_blksize
  if size <= 0:
    raise ResetIncomplete("filesystem did not report a preferred IO size")
  return size


@dataclass
class _Blob:
  """Contents shared by the baseline paths that named one regular-file inode.

  archive is the reusable source for restoration; spare is a disposable copy
  ready to rename into place. The archive is retained after installing spare.
  aliases lists managed baseline hard links, which must be reconnected after
  replacing the inode. Signatures detect modifications to our stored copies.
  """

  archive: Path
  size: int
  archive_signature: tuple
  aliases: list[Path] = field(default_factory=list)
  spare: Path | None = None
  spare_signature: tuple | None = None


@dataclass
class _Node:
  """Desired state of a baseline path, plus its last known inode signature.

  Metadata, directory children, symlink target and blob describe setup state.
  signature is refreshed after restoration: a replacement has a new inode
  and ctime, which should be accepted on the next request.
  """

  mode: int
  uid: int
  gid: int
  atime_ns: int
  mtime_ns: int
  signature: tuple
  children: frozenset[str] = frozenset()
  target: str | None = None
  blob: _Blob | None = None


class _Notifications:
  """Convert Linux filesystem events into candidate paths for reconciliation.

  inotify delivers records through one nonblocking file descriptor. A watch
  covers one directory, so prepare() registers every managed baseline directory.
  A record identifies the watch and usually a child name, e.g. creation of
  "foo" under the watched /tmp directory selects /tmp/foo.

  Records select paths to inspect; archives hold the baseline contents.
  Unsupported watches, unknown records or queue overflow require full baseline
  verification.
  """

  # Record layout and flags come from linux/inotify.h.
  # Each header contains watch descriptor, event mask, rename cookie and the
  # length of the following NUL-padded name.
  _EVENT = struct.Struct("=iIII")
  _OVERFLOW = 0x4000
  # IN_IGNORED, IN_UNMOUNT, IN_DELETE_SELF and IN_MOVE_SELF invalidate a watch.
  _LOST = 0x8000 | 0x2000 | 0x400 | 0x800
  # Watch modifications, attributes, close-after-write, both rename endpoints,
  # additions/deletions, and directory loss. ONLYDIR/DONT_FOLLOW reject symlinks
  # during watch registration.
  _MASK = (
    0x2
    | 0x4
    | 0x8
    | 0x40
    | 0x80
    | 0x100
    | 0x200
    | 0x400
    | 0x800
    | 0x2000
    | 0x01000000
    | 0x02000000
  )

  def __init__(self):
    self.fd = -1
    self.paths: dict[int, Path] = {}
    self.watched: dict[Path, int] = {}
    self.retired: set[int] = set()
    self.complete = False
    if not sys.platform.startswith("linux"):
      return
    try:
      self.libc = ctypes.CDLL(None, use_errno=True)
      self.libc.inotify_init1.argtypes = [ctypes.c_int]
      self.libc.inotify_init1.restype = ctypes.c_int
      self.libc.inotify_add_watch.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint32,
      ]
      self.libc.inotify_add_watch.restype = ctypes.c_int
      self.libc.inotify_rm_watch.argtypes = [ctypes.c_int, ctypes.c_int]
      self.libc.inotify_rm_watch.restype = ctypes.c_int
      self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
      self.complete = self.fd >= 0
    except (OSError, AttributeError):
      self.complete = False

  def add(self, path: Path):
    """Watch one baseline directory; retain any registration failure."""
    if self.fd < 0 or path in self.watched:
      return
    wd = self.libc.inotify_add_watch(self.fd, os.fsencode(path), self._MASK)
    if wd < 0:
      self.complete = False
      return
    self.paths[wd] = path
    self.watched[path] = wd
    self.retired.discard(wd)

  def drain(self, work: _Work) -> tuple[set[Path], bool]:
    """Read queued records from the nonblocking fd and return (paths, reliable).

    reliable=False requests verification of the full baseline.
    Directory moves/deletions also select the parent so restoration can repair
    the surrounding structure and install a watch on the restored directory.
    """
    dirty: set[Path] = set()
    reliable = self.complete
    if self.fd < 0:
      return dirty, False
    while True:
      work.record()
      try:
        data = os.read(self.fd, io.DEFAULT_BUFFER_SIZE)
      except BlockingIOError:
        return dirty, reliable
      if not data:
        return dirty, False
      offset = 0
      while offset < len(data):
        work.record()
        wd, mask, _cookie, length = self._EVENT.unpack_from(data, offset)
        offset += self._EVENT.size
        name = os.fsdecode(data[offset : offset + length].split(b"\0", 1)[0])
        offset += length
        if mask & self._OVERFLOW:
          reliable = False
        path = self.paths.get(wd)
        if path is None:
          if wd in self.retired:
            # Removing a watch queues IN_IGNORED. The retired set identifies
            # these expected records.
            if mask & 0x8000:
              self.retired.discard(wd)
            continue
          reliable = False
          continue
        if name:
          # A named file event selects that file, avoiding a scan of every
          # sibling in a potentially large image directory.
          dirty.add(path / name)
        else:
          dirty.add(path)
        if mask & self._LOST:
          dirty.add(path.parent)
          self.watched.pop(path, None)
          self.paths.pop(wd, None)
          if not mask & 0x8000:
            self.retired.add(wd)
            self.libc.inotify_rm_watch(self.fd, wd)

  def close(self):
    if self.fd >= 0:
      os.close(self.fd)
      self.fd = -1


class _Processes:
  """Track execution processes in a private PID namespace, including orphans.

  begin() protects identities present before a request. stop() signals newly
  observed identities and then verifies termination. UID scope and the set of
  existing identities select processes for cleanup, including those detached
  from the command's process group.

  Unprivileged execution is restricted to its UID. Root execution can change
  all of its UIDs, so it tracks all new identities in the disposable namespace.
  """

  def __init__(self, uid: int | None = None, proc: Path = Path("/proc")):
    self.uid = os.getuid() if uid is None else uid
    self.proc = proc
    self.protected: set[tuple[int, str]] = set()
    self.suspects: set[int] = set()

  def identity(self, pid: int):
    """Return (PID, start time) for a live process in scope, or None.

    Pairing PID with start time prevents a reused PID from inheriting the
    protection of an exited setup process. Exited processes return None;
    unreadable or malformed process records raise ResetIncomplete.
    """
    try:
      directory = self.proc / str(pid)
      fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
      # Split after comm, which can contain spaces. The remaining fields start
      # at stat field 3 (state); index 19 is field 22 (start time).
      if fields[0] in ("Z", "X"):
        return None
      uids = next(
        line.split()[1:]
        for line in (directory / "status").read_text().splitlines()
        if line.startswith("Uid:")
      )
      # Include real/effective/saved UIDs for setuid executables. Root execution
      # also covers new processes that have changed all their UIDs.
      if self.uid != 0 and str(self.uid) not in uids:
        return None
      return pid, fields[19]
    except FileNotFoundError:
      return None
    except OSError as exc:
      if exc.errno in (errno.ENOENT, errno.ESRCH):
        return None
      raise ResetIncomplete(f"cannot inspect process {pid}") from exc
    except (IndexError, StopIteration, ValueError) as exc:
      raise ResetIncomplete(f"cannot inspect process {pid}") from exc

  def all_pids(self, work: _Work):
    with os.scandir(self.proc) as entries:
      for entry in entries:
        work.record()
        if entry.name.isdigit():
          yield int(entry.name)

  def begin(self):
    """Remember existing process identities to preserve during this request."""
    # Infrastructure present before this command is protected by identity.
    # Previous commands must have finished cleanup before begin is permitted.
    self.protected = {
      identity
      for pid in self.all_pids(_Work())
      if (identity := self.identity(pid)) is not None
    }
    self.suspects.clear()

  def stop(self, work: _Work):
    """Signal new processes, verify survivors, and fail if reuse is unsafe.

    A pidfd pins the target when supported. The PID fallback rechecks identity
    immediately before signalling. Verification also detects new grandchildren
    spawned while the initial discovery/signalling pass was running.
    """
    # Discover candidates by enumerating PIDs in the private namespace.
    candidates = set(self.all_pids(work))
    self.suspects.clear()
    for pid in candidates:
      work.record()
      identity = self.identity(pid)
      if identity is None or identity in self.protected or pid == os.getpid():
        continue
      fd = None
      try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
          try:
            fd = os.pidfd_open(pid)
          except OSError as exc:
            if exc.errno not in (errno.ENOSYS, errno.EINVAL, errno.EPERM):
              raise
        # Recheck after opening the stable handle, or before the PID fallback.
        if self.identity(pid) != identity:
          continue
        if fd is None:
          os.kill(pid, signal.SIGKILL)
        else:
          signal.pidfd_send_signal(fd, signal.SIGKILL)
      except ProcessLookupError:
        pass
      finally:
        if fd is not None:
          os.close(fd)
    # Verify termination after signalling. Survivors require replacement.
    self.suspects = {
      pid
      for pid in self.all_pids(work)
      if (identity := self.identity(pid)) is not None
      and identity not in self.protected
      and pid != os.getpid()
    }
    if self.suspects:
      raise ResetIncomplete("execution processes survived cleanup")


@dataclass(frozen=True)
class ResetResult:
  """Request cleanup outcome and telemetry for the daemon's reuse decision.

  clean permits reuse; reclamation_pending waits for quarantine deletion;
  retire requires replacement. Elapsed time and operations are log fields.
  """

  status: str
  elapsed_ms: float
  operations: int
  quarantine_items: int
  reason: str | None = None


class SandboxReset:
  """Own the baseline and deferred recovery work for one sandbox lifetime.

  The caller supplies a real filesystem root and exclusions, calls prepare()
  before serving, then uses begin() -> execution/result collection -> finish()
  for each request. stop_processes() is also called at subprocess exit, before
  final diffs/checkpoints are collected. maintain() advances deferred work.

  The caller holds the same lock for requests and maintenance. close() releases
  open resources at shutdown.
  """

  def __init__(
    self,
    root: Path,
    state_dir: Path,
    *,
    excluded: tuple[Path, ...] = (),
    process_uid: int | None = None,
    additional_roots: tuple[Path, ...] = (),
    track_processes: bool = True,
  ):
    if root.is_symlink() or not root.is_dir():
      raise ValueError("Reset root must be a real directory")
    self.root = root.resolve(strict=True)
    if state_dir.is_symlink():
      raise ValueError("State directory cannot be a symlink")
    self.state_dir = state_dir.resolve()
    if not self.state_dir.is_relative_to(self.root):
      raise ValueError("State directory must be within the managed root")
    if process_uid is not None and process_uid < 0:
      raise ValueError("Execution UID must be nonnegative")
    self.excluded = tuple(
      dict.fromkeys(
        version for path in excluded for version in (path.absolute(), path.resolve())
      )
    ) + (self.state_dir,)
    if any(path.is_symlink() or not path.is_dir() for path in additional_roots):
      raise ValueError(
        "Additional roots must be real directories within the reset root"
      )
    self.additional_roots = tuple(
      path.resolve(strict=True) for path in additional_roots
    )
    if any(not path.is_relative_to(self.root) for path in self.additional_roots):
      raise ValueError("Additional roots must be within the reset root")
    self.nodes: dict[Path, _Node] = {}
    # Baseline device/inode keys deduplicate contents across hard-link names.
    self.blobs: dict[tuple[int, int], _Blob] = {}
    self.bytes_saved = 0
    self.notifications = _Notifications()
    self.processes = _Processes(process_uid) if track_processes else None
    self.reason: str | None = None
    self.prepared = False
    self.active = False
    self.spent_ms = 0.0
    self.operations = 0
    self.trash: deque[Iterator[None]] = deque()
    self._trash_serial = 0
    self._refill: deque[_Blob] = deque()
    self._touched_dirs: set[Path] = set()
    self._partial: tuple[_Blob, Path, Iterator[None]] | None = None
    self._root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

  @property
  def available(self):
    """Whether baseline and reclamation state permit another request.

    Requests can run during spare refills. Restoration copies from the archive
    when a spare is missing. begin() rejects active execution.
    """
    # New requests wait for quarantine deletion to finish.
    return self.prepared and self.reason is None and not self.trash

  @property
  def needs_maintenance(self):
    """Whether quarantine deletion or replacement preparation can advance."""
    return bool(self.trash or self._partial or self._refill) and self.reason is None

  def excluded_path(self, path: Path):
    """Apply caller exclusions, allowing explicitly managed subtrees.

    For example, /dev may be excluded while /dev/shm is an additional root.
    Private reset storage always remains excluded from reconciliation.
    """
    for parent in self.excluded:
      if path != parent and parent not in path.parents:
        continue
      if parent != self.state_dir and any(
        (path == root or root in path.parents) and parent in root.parents
        for root in self.additional_roots
      ):
        continue
      return True
    return False

  @contextmanager
  def parent_fd(self, path: Path):
    """Yield a parent directory fd and leaf name by walking from the pinned root.

    O_DIRECTORY and O_NOFOLLOW reject symlink ancestors. If a command has
    substituted a symlink for a directory, the open fails and cleanup retires
    the sandbox.
    """
    parts = path.relative_to(self.root).parts
    if not parts:
      raise ResetIncomplete("cannot replace the managed root")
    fd = os.dup(self._root_fd)
    try:
      for part in parts[:-1]:
        next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        os.close(fd)
        fd = next_fd
      yield fd, parts[-1]
    finally:
      os.close(fd)

  def info(self, path: Path, work: _Work):
    """Read leaf metadata with lstat semantics; return None for a missing leaf."""
    work.record()
    if path == self.root:
      return os.fstat(self._root_fd)
    with self.parent_fd(path) as (fd, name):
      try:
        return os.stat(name, dir_fd=fd, follow_symlinks=False)
      except FileNotFoundError:
        return None

  def prepare(self):
    """Capture setup state once, before accepting execution requests.

    Create private baseline/files/trash directories, record path metadata,
    archive each unique regular-file inode, and prepare same-filesystem spares.
    Watches are installed as directories are inventoried. This is the full
    startup traversal; capture or copy errors propagate to the startup caller.
    """
    if self.prepared:
      raise RuntimeError("Baseline already captured")
    if not self.state_dir.is_relative_to(self.root):
      raise ValueError("State directory must be within the managed root")
    self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    (self.state_dir / "baseline").mkdir()
    (self.state_dir / "files").mkdir()
    (self.state_dir / "trash").mkdir()
    self._state_signature = _signature(self.state_dir.lstat())[:2]
    self._state_device = self.state_dir.stat().st_dev
    # Capture explicitly managed writable roots (such as /dev/shm) separately
    # from the image's excluded virtual infrastructure.
    pending = [self.root] + list(reversed(self.additional_roots))
    while pending:
      path = pending.pop()
      if self.excluded_path(path) or path in self.nodes:
        continue
      info = path.lstat()
      node = _Node(
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_atime_ns,
        info.st_mtime_ns,
        _signature(info),
      )
      self.nodes[path] = node
      if stat.S_ISDIR(info.st_mode):
        self.notifications.add(path)
        with os.scandir(path) as entries:
          children = [
            Path(entry.path)
            for entry in entries
            if not self.excluded_path(Path(entry.path))
          ]
        node.children = frozenset(child.name for child in children)
        pending.extend(reversed(children))
      elif stat.S_ISLNK(info.st_mode):
        node.target = os.readlink(path)
      elif stat.S_ISREG(info.st_mode):
        key = (info.st_dev, info.st_ino)
        blob = self.blobs.get(key)
        if blob is None:
          # Archive file contents on disk, once per baseline inode.
          archive = self.state_dir / "baseline" / f"file-{len(self.blobs)}"
          self._copy_file(path, archive)
          archived = archive.lstat()
          if (
            archived.st_size != info.st_size
            or _signature(path.lstat()) != node.signature
          ):
            raise ResetIncomplete(f"baseline changed during capture: {path}")
          self.bytes_saved += info.st_size
          blob = self.blobs[key] = _Blob(archive, info.st_size, _signature(archived))
        blob.aliases.append(path)
        node.blob = blob
    for blob in self.blobs.values():
      if blob.size and self.nodes[blob.aliases[0]].signature[0] == self._state_device:
        self._write_spare(blob)
    self.prepared = True

  def _write_spare(self, blob: _Blob):
    """Create a disposable replacement while retaining the reusable archive."""
    # The serial supplies a fresh name after a previous spare is consumed.
    self._trash_serial += 1
    path = self.state_dir / "files" / f"copy-{self._trash_serial}"
    self._copy_file(blob.archive, path)
    blob.spare = path
    blob.spare_signature = _signature(path.lstat())

  def _copy_contents(self, source: Path, destination):
    """Copy in chunks, yielding after each chunk.

    Chunk size comes from the destination filesystem. Extent discovery skips
    sparse holes when supported; final truncation preserves logical file size.
    Maintenance resumes this iterator between requests.
    """
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
      size = os.fstat(fd).st_size
      chunk_bytes = _copy_chunk_bytes(destination.fileno())
      sparse = False
      if size and hasattr(os, "SEEK_DATA") and hasattr(os, "SEEK_HOLE"):
        try:
          os.lseek(fd, 0, os.SEEK_HOLE)
          sparse = True
        except OSError as exc:
          if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
            raise
      offset = 0
      while offset < size:
        end = size
        if sparse:
          try:
            offset = os.lseek(fd, offset, os.SEEK_DATA)
          except OSError as exc:
            if exc.errno == errno.ENXIO:
              break
            raise
          end = min(size, os.lseek(fd, offset, os.SEEK_HOLE))
          if end <= offset:
            raise ResetIncomplete("baseline extent changed while copying")
        os.lseek(fd, offset, os.SEEK_SET)
        destination.seek(offset)
        while offset < end:
          chunk = os.read(fd, min(chunk_bytes, end - offset))
          if not chunk:
            raise ResetIncomplete("baseline file changed while copying")
          destination.write(chunk)
          offset += len(chunk)
          yield
      destination.truncate(size)
    finally:
      os.close(fd)

  def _copy_steps(self, source: Path, destination: Path):
    """Create an exclusive destination and keep its stream across copy yields."""
    with destination.open("xb") as stream:
      yield from self._copy_contents(source, stream)

  def _copy_file(self, source: Path, destination: Path):
    """Consume the copy iterator synchronously during baseline preparation."""
    for _ in self._copy_steps(source, destination):
      pass

  def _state_intact(self):
    """Check that the private storage directory still has its captured identity."""
    return (
      _signature(self.state_dir.lstat())[:2] == self._state_signature
      and self.state_dir.is_dir()
      and not self.state_dir.is_symlink()
    )

  def retire(self, reason: str):
    """Preserve the first cleanup failure; the daemon schedules replacement."""
    self.reason = self.reason or reason

  def begin(self):
    """Enter a request with reusable state and protect existing processes."""
    if not self.available or self.active:
      raise ResetIncomplete(self.reason or "reclamation is still pending")
    try:
      if not self._state_intact():
        raise ResetIncomplete("private reset storage was replaced")
      if self.processes is not None:
        self.processes.begin()
    except (OSError, ResetIncomplete) as exc:
      self.retire(str(exc))
      raise ResetIncomplete(str(exc)) from exc
    self.active = True
    self.spent_ms = 0
    self.operations = 0
    self._touched_dirs.clear()

  @contextmanager
  def stage(self):
    """Accumulate time/work spent in cleanup, excluding command/result work."""
    start = time.monotonic()
    work = _Work()
    try:
      yield work
    finally:
      self.spent_ms += (time.monotonic() - start) * 1000
      self.operations += work.operations

  def stop_processes(self):
    """Stop execution-scope writers; retain failure for finish() to report.

    The early subprocess hook and the final request check share this method.
    Process cleanup failure retires the sandbox. finish() then skips filesystem
    restoration.
    """
    if not self.active or self.reason or self.processes is None:
      return
    try:
      with self.stage() as work:
        self.processes.stop(work)
    except (OSError, ResetIncomplete) as exc:
      self.retire(f"process cleanup incomplete: {exc}")

  def _metadata(self, path: Path, node: _Node, work: _Work):
    """Restore ownership, permissions and timestamps, then accept the new inode.

    Repairs change ctime and may replace the inode. Updating signature accepts
    those changes while retaining the desired setup metadata.
    """
    info = self.info(path, work)
    if path == self.root:
      if (info.st_uid, info.st_gid) != (node.uid, node.gid):
        work.record()
        os.fchown(self._root_fd, node.uid, node.gid)
      if stat.S_IMODE(info.st_mode) != stat.S_IMODE(node.mode):
        work.record()
        os.fchmod(self._root_fd, stat.S_IMODE(node.mode))
      if (info.st_mtime_ns, info.st_atime_ns) != (node.mtime_ns, node.atime_ns):
        work.record()
        os.utime(self._root_fd, ns=(node.atime_ns, node.mtime_ns))
      node.signature = _signature(self.info(path, work))
      return
    with self.parent_fd(path) as (fd, name):
      if (info.st_uid, info.st_gid) != (node.uid, node.gid):
        work.record()
        os.chown(name, node.uid, node.gid, dir_fd=fd, follow_symlinks=False)
      if not stat.S_ISLNK(node.mode) and stat.S_IMODE(info.st_mode) != stat.S_IMODE(
        node.mode
      ):
        work.record()
        leaf = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        try:
          os.fchmod(leaf, stat.S_IMODE(node.mode))
        finally:
          os.close(leaf)
      if (info.st_mtime_ns, info.st_atime_ns) != (node.mtime_ns, node.atime_ns):
        work.record()
        os.utime(
          name, ns=(node.atime_ns, node.mtime_ns), dir_fd=fd, follow_symlinks=False
        )
    node.signature = _signature(self.info(path, work))

  def _delete_tree(self, path: Path):
    """Yield during a depth-first deletion of an already quarantined tree.

    Iterator state retains directory scanners between maintenance calls.
    Closing the iterator also closes unfinished scanners.
    """
    # An explicit stack handles deeply nested directories.
    # Each yield returns control so the caller can release the execution lock.
    stack = [(path, None)]
    try:
      while stack:
        yield
        current, entries = stack[-1]
        if entries is None:
          info = current.lstat()
          if stat.S_ISDIR(info.st_mode):
            stack[-1] = current, os.scandir(current)
          else:
            current.unlink()
            stack.pop()
        else:
          try:
            entry = next(entries)
          except StopIteration:
            entries.close()
            current.rmdir()
            stack.pop()
          else:
            stack.append((Path(entry.path), None))
    finally:
      for _path, entries in stack:
        if entries is not None:
          entries.close()

  def _discard(self, path: Path, work: _Work):
    """Unlink a leaf or rename a directory into quarantine.

    A same-filesystem rename removes a new tree from its visible location.
    Recursive deletion is deferred, and pending quarantine blocks further
    execution until reclaimed. Cross-filesystem moves require retirement.
    """
    info = self.info(path, work)
    if info is None:
      return
    with self.parent_fd(path) as (fd, name):
      work.record()
      if not stat.S_ISDIR(info.st_mode):
        os.unlink(name, dir_fd=fd)
        return
      self._trash_serial += 1
      destination = self.state_dir / "trash" / str(self._trash_serial)
      try:
        os.rename(name, destination, src_dir_fd=fd)
      except OSError as exc:
        if exc.errno == errno.EXDEV:
          raise ResetIncomplete("cannot quarantine across filesystems") from exc
        raise
      self.trash.append(self._delete_tree(destination))

  def _ensure_directory(self, path: Path, work: _Work):
    """Repair baseline ancestors before resolving any child through them."""
    if path == self.root or path in self.additional_roots:
      return
    node = self.nodes.get(path)
    if node is None or not stat.S_ISDIR(node.mode):
      raise ResetIncomplete(f"missing baseline parent: {path}")
    self._ensure_directory(path.parent, work)
    info = self.info(path, work)
    if info is not None and not stat.S_ISDIR(info.st_mode):
      self._discard(path, work)
      info = None
    if info is None:
      with self.parent_fd(path) as (fd, name):
        work.record()
        os.mkdir(name, stat.S_IMODE(node.mode), dir_fd=fd)
      self.notifications.add(path)
      self._touched_dirs.update((path, path.parent))

  def _restore_file(self, path: Path, node: _Node, work: _Work):
    """Install baseline contents by rename and reconnect managed hard links.

    A prepared spare avoids copying contents in the reset window. If absent,
    copy the archive beside the destination before renaming. The old inode
    retains the command's contents, including through any new hard link under
    an excluded workspace such as /app.
    """
    blob = node.blob
    if _signature(blob.archive.lstat()) != blob.archive_signature:
      raise ResetIncomplete("baseline archive was modified")
    self._ensure_directory(path.parent, work)
    spare = blob.spare
    if spare is not None and _signature(spare.lstat()) != blob.spare_signature:
      raise ResetIncomplete("prepared replacement was modified")
    if spare is None:
      # Exhausted/cross-device reserves copy from the archived baseline.
      self._trash_serial += 1
      spare = path.parent / f".daemon-replacement-{self._trash_serial}"
      with self.parent_fd(spare) as (fd, name):
        work.record()
        out = os.open(
          name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd
        )
        with os.fdopen(out, "wb") as stream:
          for _ in self._copy_contents(blob.archive, stream):
            work.record()
    current = self.info(path, work)
    if current is not None and not stat.S_ISREG(current.st_mode):
      self._discard(path, work)
    with self.parent_fd(path) as (fd, name):
      work.record()
      os.replace(spare, name, dst_dir_fd=fd)
    blob.spare = None
    blob.spare_signature = None
    if blob.size:
      self._refill.append(blob)
    self._metadata(path, node, work)
    # Reconnect baseline aliases to the replacement inode.
    for alias in blob.aliases:
      work.record()
      self._touched_dirs.add(alias.parent)
      if alias == path:
        continue
      self._ensure_directory(alias.parent, work)
      self._discard(alias, work)
      with self.parent_fd(path) as (source, source_name):
        with self.parent_fd(alias) as (destination, name):
          os.link(
            source_name,
            name,
            src_dir_fd=source,
            dst_dir_fd=destination,
            follow_symlinks=False,
          )
    signature = _signature(self.info(path, work))
    for alias in blob.aliases:
      self.nodes[alias].signature = signature

  def _restore(self, path: Path, work: _Work, pending: set[Path]):
    """Reconcile one candidate path and enqueue any newly discovered children.

    Absent baseline paths are additions to discard. Baseline directories are
    compared by immediate entry names; deeper repair is scheduled separately.
    Unchanged regular files can be skipped using their last known signature.
    """
    if self.excluded_path(path) or not path.is_relative_to(self.root):
      return
    node = self.nodes.get(path)
    self._touched_dirs.add(path.parent)
    if node is None:
      self._discard(path, work)
      return
    if path != self.root and path not in self.additional_roots:
      self._ensure_directory(path.parent, work)
    info = self.info(path, work)
    if stat.S_ISDIR(node.mode):
      replaced = (
        info is None
        or not stat.S_ISDIR(info.st_mode)
        or _signature(info)[:2] != node.signature[:2]
      )
      if path != self.root:
        self._ensure_directory(path, work)
      fd = os.dup(self._root_fd) if path == self.root else None
      try:
        if fd is None:
          with self.parent_fd(path) as (parent, name):
            fd = os.open(
              name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
        with os.scandir(fd) as entries:
          actual = set()
          for entry in entries:
            work.record()
            child = path / entry.name
            if not self.excluded_path(child):
              actual.add(entry.name)
        for name in actual ^ node.children:
          work.record()
          pending.add(path / name)
        if replaced:
          # Verify every baseline child after directory replacement, including
          # children whose names match.
          for name in node.children:
            work.record()
            pending.add(path / name)
        self.notifications.add(path)
      finally:
        if fd is not None:
          os.close(fd)
      self._touched_dirs.add(path)
    elif info is not None and _signature(info) == node.signature:
      return
    elif stat.S_ISREG(node.mode):
      self._restore_file(path, node, work)
    elif stat.S_ISLNK(node.mode):
      if info is not None:
        self._discard(path, work)
      with self.parent_fd(path) as (fd, name):
        work.record()
        os.symlink(node.target, name, dir_fd=fd)
      self._metadata(path, node, work)
    else:
      raise ResetIncomplete(f"changed special file cannot be restored: {path}")

  def finish(self) -> ResetResult:
    """End a request after its result has been collected, returning reuse status.

    Verify processes stopped, drain inotify, and repair candidates with parents
    before children. Lost notification coverage selects the whole inventory.
    Restore directory metadata last because child repairs modify parent times.
    Failures are retained as retire; the caller handles HTTP/restart behavior.
    """
    if not self.active:
      raise RuntimeError("No command is active")
    self.stop_processes()
    try:
      if self.reason is None:
        with self.stage() as work:
          if not self._state_intact():
            raise ResetIncomplete("private reset storage was replaced")
          pending, reliable = self.notifications.drain(work)
          if not reliable:
            # Failed notification coverage requires checking all baseline paths.
            pending.update(self.nodes)
          # The heap orders candidates by depth, including children discovered
          # during repair.
          queue = [(len(path.parts), str(path), path) for path in pending]
          heapq.heapify(queue)
          while queue:
            work.record()
            _depth, _name, path = heapq.heappop(queue)
            pending.remove(path)
            discovered: set[Path] = set()
            self._restore(path, work, discovered)
            for child in discovered - pending:
              pending.add(child)
              heapq.heappush(queue, (len(child.parts), str(child), child))
          for path in sorted(
            self._touched_dirs, key=lambda item: len(item.parts), reverse=True
          ):
            node = self.nodes.get(path)
            if node is not None and stat.S_ISDIR(node.mode):
              self._metadata(path, node, work)
          # Discard events generated by these repairs before the next request.
          self.notifications.drain(work)
    except (OSError, ResetIncomplete) as exc:
      self.retire(f"filesystem cleanup incomplete: {exc}")
    finally:
      self.active = False
    return ResetResult(
      "retire" if self.reason else ("reclamation_pending" if self.trash else "clean"),
      self.spent_ms,
      self.operations,
      len(self.trash),
      self.reason,
    )

  def maintain(self):
    """Advance deferred deletion/copy work while holding the execution lock.

    Resume one iterator step, then return so the caller can release the lock.
    Quarantine is reclaimed first because it blocks reuse. A partially copied
    spare becomes usable after its iterator finishes. Until then, reset copies
    the archive synchronously.
    """
    if self.active or self.reason:
      return
    try:
      if not self._state_intact():
        raise ResetIncomplete("private reset storage was replaced")
      if self.trash:
        try:
          next(self.trash[0])
        except StopIteration:
          self.trash.popleft()
        return
      if self._partial or self._refill:
        if self._partial is None:
          blob = self._refill.popleft()
          if (
            blob.spare is not None
            or self.nodes[blob.aliases[0]].signature[0] != self._state_device
          ):
            return
          self._trash_serial += 1
          path = self.state_dir / "files" / f"copy-{self._trash_serial}"
          self._partial = blob, path, self._copy_steps(blob.archive, path)
        blob, path, steps = self._partial
        try:
          next(steps)
        except StopIteration:
          blob.spare = path
          blob.spare_signature = _signature(path.lstat())
          self._partial = None
    except ResetIncomplete as exc:
      self.retire(str(exc))
    except OSError as exc:
      self.retire(f"maintenance failed: {exc}")

  def close(self):
    """Close paused iterators, watches and the root fd; retain on-disk storage."""
    if self._partial:
      self._partial[2].close()
    for iterator in self.trash:
      iterator.close()
    self.notifications.close()
    os.close(self._root_fd)

"""One queue for the GPU: every process that runs kernels on this machine takes a ticket first, and exactly one
holds the GPU at a time.

Several CUDA processes computing at once on the display GPU caused NVIDIA driver resets (2026-10-19/20): the stream
and the batch, the stream's probe and its llama.cpp screen, and loads beside running work. Memory caps and partial
locks each closed one gap. This is the single mechanism every GPU client uses:

  with gpu_turn.turn("probe"):            # the live stream: priority 0, the default
      ... one model call: a forward pass, a load, one screen request ...
  with gpu_turn.turn("auto2", priority=1):  # a batch: always behind any waiting stream ticket
      ... one batch ...

- **Queue:** a ticket is a file `ticket-<priority>-<time_ns>-<pid>-<tag>`. The live ticket with the lowest
  (priority, time) is next. First come, first served within a priority, and the stream always goes before the batch.
- **Exclusive:** the head also takes an OS file lock on `gpu.lock` (msvcrt on Windows, flock elsewhere), so two
  holders are impossible. The OS drops the lock if the holder dies, and tickets of dead processes are removed.
- **Re-entrant within a process:** nested or concurrent turns of the same process (e.g. a server's request threads)
  share one hold.
- **Short holds:** hold one model call per turn, not a whole analysis, so others interleave.
- **Visible:** the holder writes `holder.json` (pid, tag, priority, since), and every finished turn appends
  {pid, tag, priority, start, end} to `turns.log`, so an auditor can match the processes actually using the GPU against
  turn intervals.

Kept for existing callers:
- `foreground(tag)`: the same as `turn(tag, priority=0)`.
- `background_tick(tag)`: call before each batch. It keeps the turn while no stream ticket waits; otherwise it
  synchronises, steps back into the queue at priority 1 and returns the seconds it waited.
- `release()`: drops the background's turn.

The directory is shared by every checkout on the machine: BONSAI_GPU_TURN (environment or .env), else
%USERPROFILE%/bonsai-gpu-turn on Windows (never AppData: see _dir) and ~/.cache/bonsai-gpu-turn elsewhere.
"""
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

POLL = 0.02


def _setting(name):
    """Like env._setting, without importing torch (a waiting caller must not pay for that)."""
    v = os.environ.get(name)
    env = Path(__file__).resolve().parent.parent / ".env"
    if not v and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "="):
                v = line.split("=", 1)[1].strip().strip('"')
    return v or None


def _dir():
    d = _setting("BONSAI_GPU_TURN")
    if d:
        p = Path(d)
    elif sys.platform == "win32":
        # NOT under AppData: processes started inside an MSIX-packaged app (e.g. the Claude desktop app) get AppData
        # writes redirected to a private copy, so their tickets and lock were invisible to everyone outside the
        # package (2026-10-20). The user profile root is not redirected; a lock there excludes across the boundary.
        p = Path(os.environ.get("USERPROFILE", Path.home())) / "bonsai-gpu-turn"
    else:
        p = Path.home() / ".cache" / "bonsai-gpu-turn"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _alive(pid):
    if pid == os.getpid():
        return True
    if sys.platform == "win32":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _sync():
    torch = sys.modules.get("torch")  # only a process that already uses torch can have kernels in flight
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except Exception:
        pass


def _tickets():
    """Live tickets as (priority, time_ns, pid, path), oldest first within a priority; dead owners' tickets removed."""
    out = []
    for f in _dir().glob("ticket-*"):
        try:
            _, prio, tns, pid, _tag = f.name.split("-", 4)
            prio, tns, pid = int(prio), int(tns), int(pid)
        except ValueError:
            continue
        if _alive(pid):
            out.append((prio, tns, pid, f))
        else:
            f.unlink(missing_ok=True)
    return sorted(out)


class _OSLock:
    """An exclusive lock on one byte of gpu.lock, released by the OS if the process dies."""

    def __init__(self):
        self.fd = None

    def try_acquire(self):
        fd = os.open(_dir() / "gpu.lock", os.O_RDWR | os.O_CREAT)
        try:
            if sys.platform == "win32":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self.fd = fd
        return True

    def release(self):
        if self.fd is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None


_state = {"depth": 0, "prio": None, "ticket": None, "lock": _OSLock()}
_plock = threading.Lock()   # guards _state
_acq = threading.Lock()     # one acquisition at a time within the process


def _acquire(tag, prio, timeout):
    """Queue, wait to be the head, take the OS lock. Returns the seconds waited."""
    t0 = time.time()
    ticket = _dir() / f"ticket-{prio}-{time.time_ns()}-{os.getpid()}-{tag}"
    ticket.write_text("")
    warned = False
    while True:
        q = _tickets()
        head = q[0][3] if q else None
        if head is not None and head.name == ticket.name and _state["lock"].try_acquire():
            break
        if timeout is not None and time.time() - t0 > timeout and not warned:
            print(f"gpu_turn: '{tag}' has waited {timeout:.0f} s for the GPU (holder: {holder()})", file=sys.stderr)
            warned = True
        time.sleep(POLL)
    _state.update(prio=prio, ticket=ticket, tag=tag, start=time.time())
    try:
        (_dir() / "holder.json").write_text(json.dumps({"pid": os.getpid(), "tag": tag, "priority": prio, "since": time.time()}))
    except OSError:
        pass
    return time.time() - t0


def _release_hold():
    _sync()  # nothing of ours is in flight when the next holder starts
    try:
        hj = _dir() / "holder.json"
        if hj.exists() and json.loads(hj.read_text() or "{}").get("pid") == os.getpid():
            hj.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass
    _state["lock"].release()
    end = time.time()
    if _state["ticket"] is not None:
        _state["ticket"].unlink(missing_ok=True)
    try:  # one line per turn, machine-wide, so an auditor can match GPU use to turn intervals, not to snapshots
        with open(_dir() / "turns.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "tag": _state.get("tag"), "priority": _state["prio"],
                                "start": round(_state.get("start") or end, 3), "end": round(end, 3)}) + "\n")
    except OSError:
        pass
    _state.update(prio=None, ticket=None, tag=None, start=None)


def _enter(tag, prio, timeout):
    with _plock:
        if _state["depth"] > 0:
            _state["depth"] += 1
            return 0.0
    with _acq:
        with _plock:
            if _state["depth"] > 0:
                _state["depth"] += 1
                return 0.0
        waited = _acquire(tag, prio, timeout)
        with _plock:
            _state["depth"] = 1
        return waited


def _exit():
    with _plock:
        _state["depth"] -= 1
        if _state["depth"] == 0:
            _release_hold()


@contextmanager
def turn(tag="gpu", priority=0, timeout=60.0):
    """Hold the GPU for the block: one model call. Priority 0 (the live stream) always goes before 1 (a batch).
    `timeout` only prints a warning; the turn still waits, because computing beside the holder is what resets the
    driver."""
    _enter(tag, priority, timeout)
    try:
        yield
    finally:
        _exit()


def foreground(tag="fg", timeout=60.0):
    return turn(tag, 0, timeout)


_bg = {"held": False}


def background_tick(tag="bg"):
    """Call before every batch of a background job. Keeps the GPU while no stream (priority-0) ticket waits;
    otherwise steps back into the queue behind it. Returns the seconds spent waiting."""
    if _bg["held"]:
        if not any(p == 0 and pid != os.getpid() for p, _, pid, _ in _tickets()):
            return 0.0
        _bg["held"] = False
        _exit()
    waited = _enter(tag, 1, None)
    _bg["held"] = True
    return waited


def release():
    """A background that stops, or pauses for CPU work, gives the GPU up."""
    if _bg["held"]:
        _bg["held"] = False
        _exit()


def holder():
    """Who holds the GPU now ({pid, tag, priority, since}), or None."""
    try:
        h = json.loads((_dir() / "holder.json").read_text() or "{}")
        return h if h and _alive(int(h.get("pid", 0))) else None
    except (OSError, ValueError):
        return None

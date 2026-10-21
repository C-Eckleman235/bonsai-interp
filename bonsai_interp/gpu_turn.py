"""Take turns on a shared GPU: only one process computes at a time, and the foreground always goes first.

Two CUDA processes computing at once on the display GPU caused NVIDIA driver resets (2026-10-19/20), even with
memory caps. Both can keep their model loaded; what they must not do is run kernels at the same time.

  foreground (the stream's probe, an interactive command):
      with gpu_turn.foreground("stream"):
          ... GPU work ...
  background (a batch job), once before every batch:
      gpu_turn.background_tick("auto2")      # yields at once if a foreground wants the GPU

The background holds a busy file while it computes. A foreground writes a want file, then waits until no busy file
is left and no other process holds the foreground lock (fg.lock): foregrounds of different processes take turns too. The background checks for want files before each batch: it finishes its in-flight kernels
(cuda.synchronize), removes its busy file, waits until the wants are gone, and takes the GPU back. So a foreground
waits at most one background batch. Files carry the owner's pid; a dead owner's files are removed.

The directory is shared by every checkout on the machine: BONSAI_GPU_TURN (environment or .env), else
%LOCALAPPDATA%/bonsai-gpu-turn on Windows and ~/.cache/bonsai-gpu-turn elsewhere.
"""
import atexit
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

POLL = 0.05


def _setting(name):
    """Like env._setting, without importing torch (a foreground must not pay for that while it waits)."""
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
        p = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "bonsai-gpu-turn"
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


def _live(kind):
    """Live files of one kind ('want' or 'busy'); a dead owner's files are removed."""
    out = []
    for f in _dir().glob(f"{kind}-*"):
        try:
            pid = int(f.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if _alive(pid):
            out.append(f)
        else:
            f.unlink(missing_ok=True)
    return out


def _sync():
    torch = sys.modules.get("torch")  # only a process that already uses torch can have kernels in flight
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except Exception:
        pass


def _mine(kind, tag):
    return _dir() / f"{kind}-{os.getpid()}-{tag}"


_fg_depth = 0  # re-entrant within one process: nested or concurrent foregrounds of the same process share its lock
_fg_mutex = threading.Lock()


def _fg_lock_path():
    return _dir() / "fg.lock"


def _try_fg_lock():
    """Exclusive between processes: one foreground process computes at a time (2026-10-20: the stream's probe and its
    llama.cpp screen computing together preceded a driver reset). A dead owner's lock is removed."""
    p = _fg_lock_path()
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            owner = int(p.read_text().strip() or 0)
        except (OSError, ValueError):
            return False
        if owner == os.getpid():
            return True
        if owner and not _alive(owner):
            p.unlink(missing_ok=True)
        return False


@contextmanager
def foreground(tag="fg", timeout=300.0):
    """Wait until no background holds the GPU and no other process's foreground is computing, keep the GPU for the
    block, then release. After `timeout` seconds of waiting it goes ahead anyway, with a warning: a foreground (the
    live stream) must not hang forever. Hold it per model call (a forward pass, a screen request), not per analysis,
    so other foregrounds interleave in short turns."""
    global _fg_depth
    want = _mine("want", tag)
    want.write_text(str(time.time()))
    t0, locked = time.time(), False
    try:
        while True:
            with _fg_mutex:
                if _fg_depth > 0 or _try_fg_lock():
                    _fg_depth += 1
                    locked = True
            if locked:
                break
            if time.time() - t0 > timeout:
                print(f"gpu_turn: another foreground still computes after {timeout:.0f} s; going ahead", file=sys.stderr)
                break
            time.sleep(POLL)
        while [f for f in _live("busy") if f.name != _mine("busy", tag).name]:
            if time.time() - t0 > timeout:
                print(f"gpu_turn: a background still holds the GPU after {timeout:.0f} s; going ahead", file=sys.stderr)
                break
            time.sleep(POLL)
        yield
    finally:
        _sync()
        want.unlink(missing_ok=True)
        if locked:
            with _fg_mutex:
                _fg_depth -= 1
                if _fg_depth == 0:
                    _fg_lock_path().unlink(missing_ok=True)


_held = None


def background_tick(tag="bg"):
    """Call before every batch. Takes the GPU on the first call; yields to any foreground that wants it.
    Returns the seconds spent yielding (0.0 when nobody was waiting)."""
    global _held
    busy = _mine("busy", tag)
    if _held is None:
        _held = busy
        atexit.register(lambda: busy.unlink(missing_ok=True))
    if not _live("want"):
        if not busy.exists():
            busy.write_text(str(time.time()))
        return 0.0
    t0 = time.time()
    _sync()  # nothing of ours is in flight before we let go
    busy.unlink(missing_ok=True)
    while True:
        while _live("want"):
            time.sleep(POLL)
        busy.write_text(str(time.time()))
        if not _live("want"):  # a foreground that arrived in between goes first
            return time.time() - t0
        busy.unlink(missing_ok=True)


def release():
    """A background that stops (or pauses for CPU work) gives the GPU up."""
    global _held
    if _held is not None:
        _sync()
        _held.unlink(missing_ok=True)
        _held = None

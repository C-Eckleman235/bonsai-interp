"""gpu_turn: one queue, one holder. A background yields to the stream within one batch; foregrounds of different
processes never overlap; a dead holder's lock is freed by the OS."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BG = r"""
import sys, time
from bonsai_interp import gpu_turn
log = open(sys.argv[1], "a")
end = time.time() + float(sys.argv[2])
while time.time() < end:
    gpu_turn.background_tick("test")
    log.write(f"B {time.time():.4f} start" + chr(10)); log.flush()
    time.sleep(0.2)                       # one 'batch'
    log.write(f"B {time.time():.4f} end" + chr(10)); log.flush()
gpu_turn.release()
"""

FG = r"""
import sys, time
from bonsai_interp import gpu_turn
log = open(sys.argv[1], "a")
who, n, hold = sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
for _ in range(n):
    time.sleep(0.05)
    t = time.time()
    with gpu_turn.turn(who):
        log.write(f"{who} {time.time():.4f} start wait={time.time() - t:.3f}" + chr(10)); log.flush()
        time.sleep(hold)
        log.write(f"{who} {time.time():.4f} end" + chr(10)); log.flush()
"""

HOLD_AND_DIE = r"""
import os, sys, time
from bonsai_interp import gpu_turn
cm = gpu_turn.turn("doomed"); cm.__enter__()
open(sys.argv[1], "w").write("held")
time.sleep(30)
"""


def _env(d):
    return {**os.environ, "BONSAI_GPU_TURN": str(Path(d) / "turn"), "BONSAI_DEVICE": "cpu", "PYTHONPATH": str(ROOT)}


def _spans(lines, who):
    out, cur = [], None
    for e in lines:
        if e[0] != who:
            continue
        if e[2] == "start":
            cur = float(e[1])
        elif cur is not None:
            out.append((cur, float(e[1])))
            cur = None
    return out


def _disjoint(a, b):
    for a0, a1 in a:
        for b0, b1 in b:
            assert a1 <= b0 + 1e-3 or b1 <= a0 + 1e-3, (a0, a1, b0, b1)


def test_stream_vs_batch():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.txt"
        bg = subprocess.Popen([sys.executable, "-c", BG, str(log), "4"], env=_env(d))
        time.sleep(1.0)
        fg = subprocess.run([sys.executable, "-c", FG, str(log), "F", "3", "0.4"], env=_env(d), timeout=60)
        bg.wait(timeout=60)
        assert fg.returncode == 0 and bg.returncode == 0
        ev = [l.split() for l in log.read_text().splitlines()]
        F, B = _spans(ev, "F"), _spans(ev, "B")
        assert len(F) == 3 and len(B) > 5
        _disjoint(F, B)
        waits = [float(e[3].split("=")[1]) for e in ev if e[0] == "F" and e[2] == "start"]
        assert max(waits) < 0.6, waits  # at most about one background batch


def test_foregrounds_exclusive():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.txt"
        a = subprocess.Popen([sys.executable, "-c", FG, str(log), "A", "5", "0.12"], env=_env(d))
        b = subprocess.Popen([sys.executable, "-c", FG, str(log), "S", "5", "0.12"], env=_env(d))
        assert a.wait(timeout=60) == 0 and b.wait(timeout=60) == 0
        ev = [l.split() for l in log.read_text().splitlines()]
        A, S = _spans(ev, "A"), _spans(ev, "S")
        assert len(A) == 5 and len(S) == 5
        _disjoint(A, S)
        assert not list((Path(d) / "turn").glob("ticket-*"))


def test_dead_holder_frees_the_gpu():
    with tempfile.TemporaryDirectory() as d:
        flag = Path(d) / "held.txt"
        p = subprocess.Popen([sys.executable, "-c", HOLD_AND_DIE, str(flag)], env=_env(d))
        for _ in range(200):
            if flag.exists():
                break
            time.sleep(0.05)
        p.kill()
        p.wait(timeout=10)
        log = Path(d) / "log.txt"
        t = time.time()
        r = subprocess.run([sys.executable, "-c", FG, str(log), "N", "1", "0.01"], env=_env(d), timeout=60)
        assert r.returncode == 0 and time.time() - t < 20


if __name__ == "__main__":
    test_stream_vs_batch()
    test_foregrounds_exclusive()
    test_dead_holder_frees_the_gpu()
    print("ok")

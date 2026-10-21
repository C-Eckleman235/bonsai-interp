"""gpu_turn: a background yields to a foreground within one batch, and never computes while it holds the GPU."""
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
    log.write(f"B {time.time():.4f} start\n"); log.flush()
    time.sleep(0.2)                       # one 'batch'
    log.write(f"B {time.time():.4f} end\n"); log.flush()
gpu_turn.release()
"""

FG = r"""
import sys, time
from bonsai_interp import gpu_turn
log = open(sys.argv[1], "a")
for _ in range(3):
    time.sleep(0.5)
    t = time.time()
    with gpu_turn.foreground("test"):
        log.write(f"F {time.time():.4f} start wait={time.time() - t:.3f}\n"); log.flush()
        time.sleep(0.4)
        log.write(f"F {time.time():.4f} end\n"); log.flush()
"""


def test_turns():
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "BONSAI_GPU_TURN": str(Path(d) / "turn"), "BONSAI_DEVICE": "cpu", "PYTHONPATH": str(ROOT)}
        log = Path(d) / "log.txt"
        bg = subprocess.Popen([sys.executable, "-c", BG, str(log), "4"], env=env)
        time.sleep(1.0)
        fg = subprocess.run([sys.executable, "-c", FG, str(log)], env=env, timeout=60)
        bg.wait(timeout=60)
        assert fg.returncode == 0 and bg.returncode == 0
        ev = [l.split() for l in log.read_text().splitlines()]
        # intervals of each side; no background interval may overlap a foreground interval
        def spans(who):
            out, cur = [], None
            for e in ev:
                if e[0] != who:
                    continue
                if e[2] == "start":
                    cur = float(e[1])
                elif cur is not None:
                    out.append((cur, float(e[1])))
                    cur = None
            return out
        F, B = spans("F"), spans("B")
        assert len(F) == 3 and len(B) > 5
        for a0, a1 in F:
            for b0, b1 in B:
                assert b1 <= a0 + 1e-3 or b0 >= a1 - 1e-3, (a0, a1, b0, b1)
        waits = [float(e[3].split("=")[1]) for e in ev if e[0] == "F" and e[2] == "start"]
        assert max(waits) < 0.6, waits  # at most about one background batch


FG2 = r"""
import sys, time
from bonsai_interp import gpu_turn
log = open(sys.argv[1], "a")
for _ in range(4):
    with gpu_turn.foreground(sys.argv[2]):
        log.write(f"{sys.argv[2]} {time.time():.4f} start" + chr(10)); log.flush()
        time.sleep(0.15)
        log.write(f"{sys.argv[2]} {time.time():.4f} end" + chr(10)); log.flush()
    time.sleep(0.02)
"""


def test_foregrounds_exclusive():
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "BONSAI_GPU_TURN": str(Path(d) / "turn"), "BONSAI_DEVICE": "cpu", "PYTHONPATH": str(ROOT)}
        log = Path(d) / "log.txt"
        a = subprocess.Popen([sys.executable, "-c", FG2, str(log), "A"], env=env)
        b = subprocess.Popen([sys.executable, "-c", FG2, str(log), "B"], env=env)
        assert a.wait(timeout=60) == 0 and b.wait(timeout=60) == 0
        ev = [l.split() for l in log.read_text().splitlines()]
        spans = {w: [] for w in "AB"}
        cur = {}
        for who, t, kind in ev:
            if kind == "start":
                cur[who] = float(t)
            else:
                spans[who].append((cur.pop(who), float(t)))
        assert len(spans["A"]) == 4 and len(spans["B"]) == 4
        for a0, a1 in spans["A"]:
            for b0, b1 in spans["B"]:
                assert a1 <= b0 + 1e-3 or b1 <= a0 + 1e-3, (a0, a1, b0, b1)
        assert not (Path(d) / "turn" / "fg.lock").exists()


if __name__ == "__main__":
    test_turns()
    test_foregrounds_exclusive()
    print("ok")

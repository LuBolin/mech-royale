"""Check sim/geom.py against answers from the official client geometry (run in the browser on the
same seeded queries). Usage: .venv/Scripts/python sim/test_geom.py <ref.txt>"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geom import NOHIT, Geom  # noqa: E402

tok = open(sys.argv[1]).read().split()
lab = {t: i for i, t in enumerate(tok) if t in ("S", "V", "H", "M")}
S = [float(v) for v in tok[lab["S"] + 1:lab["V"]]]
V = tok[lab["V"] + 1:lab["H"]]
H = [float(v) for v in tok[lab["H"] + 1:lab["M"]]]
M = [list(map(float, v.split(","))) for v in tok[lab["M"] + 1:]]
N = len(S)

G = Geom(os.path.join(os.path.dirname(HERE), "kit", "arena"), step_up=1.5)
G.rules[1] = 1e9  # the client has no slope rule in moveAndSlide; compare the pure port
G.rules[4] = 1.0  # trunk collision, as in the client
x0, x1, y0, y1 = G.bounds
seed = 777


def rnd():
    global seed
    seed = (seed * 16807) % 2147483647
    return seed / 2147483647


X = lambda: x0 + 40 + rnd() * (x1 - x0 - 80)  # noqa: E731
Y = lambda: y0 + 40 + rnd() * (y1 - y0 - 80)  # noqa: E731
bad = {"surface": 0, "sight": 0, "transmittance": 0, "shot": 0, "move": 0}
for i in range(N):
    x, y = X(), Y()
    if abs(G.surface(x, y) - S[i]) > 1e-3:
        bad["surface"] += 1
for i in range(N):
    ax, ay, a, d = X(), Y(), rnd() * 6.283, 5 + rnd() * 190
    bx, by = ax + math.cos(a) * d, ay + math.sin(a) * d
    az = G.surface(ax, ay) + 1 + rnd() * 20
    bz = G.surface(bx, by) + 1 + rnd() * 20
    vis, tr = V[i].split(":")
    if G.sight((ax, ay, az), (bx, by, bz)) != (vis == "1"):
        bad["sight"] += 1
    if vis == "1" or float(tr) > 0:  # transmittance is only computed when no hard blocker
        from geom import _canopy
        t = math.exp(-G.rules[2] * _canopy(G.W, ax, ay, az, bx, by, bz))
        if abs(t - float(tr)) > 1e-3:
            bad["transmittance"] += 1
    t = G.ray((ax, ay, az), (bx, by, bz), False)
    if not ((t == NOHIT and H[i] == 2) or abs(t - H[i]) < 1e-3):
        bad["shot"] += 1
rows = []
for i in range(N):
    x, y = X(), Y()
    z = G.surface(x, y)
    a, d = rnd() * 6.283, rnd() * 3
    dz = (0, -0.5, 1.5)[i % 3]
    rows.append((x, y, z, 2.5, 9, math.cos(a) * d, math.sin(a) * d, dz, 1.0))
out = G.move_many(np.array(rows))
for o, r in zip(out, M):
    if abs(o[0] - r[0]) > 1e-3 or abs(o[1] - r[1]) > 1e-3 or abs(o[2] - r[2]) > 1e-3 or (o[3] > 0.5) != (r[3] > 0.5):
        bad["move"] += 1
print(f"{N} queries per type; mismatches:", bad)
A = np.array([(X(), Y(), 10.0) for _ in range(2000)])
B = A + np.array([(math.cos(a) * 150, math.sin(a) * 150, 0) for a in np.linspace(0, 6.28, 2000)])
G.sight_many(A[:5], B[:5])  # compile
t0 = time.time()
G.sight_many(A, B)
print(f"speed: {(time.time() - t0) / 2000 * 1e6:.1f} us per 150 m sight line")
R = np.array(rows * 10)
G.move_many(R[:5])
t0 = time.time()
G.move_many(R)
print(f"speed: {(time.time() - t0) / len(R) * 1e6:.1f} us per move-and-slide")
assert sum(bad.values()) == 0

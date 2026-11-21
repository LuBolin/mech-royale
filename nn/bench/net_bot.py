"""Latency stand-in: the real bot plus the proposed net (random weights) and its feature extraction each tick.

Run inside the bot image with the entrypoint overridden; the kit's check.py measures end-to-end answer times.
NET_SCALE (env, default 1) multiplies every hidden width. Per-step timings go to stderr at exit.
"""
import atexit
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/bot")
import mechfight as mf  # noqa: E402
from brain import Bot  # noqa: E402

S = int(os.environ.get("NET_SCALE", "1"))
rng = np.random.default_rng(0)
P = {}


def lin(name, i, o):
    P[name + ".w"] = (rng.standard_normal((i, o)) / math.sqrt(i)).astype(np.float32)
    P[name + ".b"] = np.zeros(o, np.float32)


A, E, e, C, H = 128 * S, 64 * S, 32 * S, 256 * S, 128 * S  # self, mech, small-entity, core, GRU widths
lin("self1", 95, A); lin("self2", A, A)
lin("mech1", 25, E); lin("mech2", E, E)
for n, d in (("proj", 8), ("loot", 10), ("drop", 6)):
    lin(n + "1", d, e); lin(n + "2", e, e)
lin("evt", 30, e); lin("rays", 32, e)
lin("core", A + 2 * E + 8 * e, C)
lin("gru_x", C, 3 * H); lin("gru_h", H, 3 * H)
lin("q_tgt", H, E); lin("k_tgt", E, E)
lin("move1", H, 64); lin("move2", 64, 102)
lin("fire", H, 2); lin("loot_h", H, 3)
lin("q_up", H, 32); P["up_emb"] = rng.standard_normal((22, 32)).astype(np.float32)
lin("val1", H, 64); lin("val2", 64, 1)
N_PARAMS = sum(v.size for v in P.values())


def L(n, x): return x @ P[n + ".w"] + P[n + ".b"]
def relu(x): return np.maximum(x, 0)
def mlp2(a, x): return relu(L(a + "2", relu(L(a + "1", x))))
def sig(x): return 1 / (1 + np.exp(-x))


def forward(inp, h):
    s = mlp2("self", inp["self"]); m = mlp2("mech", inp["mechs"])
    pr = mlp2("proj", inp["proj"]); lo = mlp2("loot", inp["loot"]); dr = mlp2("drop", inp["drops"])
    z = np.concatenate([s, m.max(0), m.mean(0), pr.max(0), pr.mean(0), lo.max(0), lo.mean(0), dr.max(0),
                        dr.mean(0), relu(L("evt", inp["evt"])), relu(L("rays", inp["rays"]))])
    c = relu(L("core", z)); gx, gh = L("gru_x", c), L("gru_h", h)
    rz = sig(gx[:2 * H] + gh[:2 * H])
    n = np.tanh(gx[2 * H:] + rz[:H] * gh[2 * H:]); h = (1 - rz[H:]) * n + rz[H:] * h
    out = (np.append(L("k_tgt", m) @ L("q_tgt", h), 0).argmax(), L("move2", relu(L("move1", h))).argmax(),
           sig(L("fire", h)) > .5, L("loot_h", h).argmax(), (P["up_emb"][inp["offer"]] @ L("q_up", h)).argmax(),
           L("val2", relu(L("val1", h))))
    return out, h


def features(W, obs):
    x, y, z = obs["self"]["pos"]
    mechs = sorted(obs["mechs"], key=lambda q: (q["pos"][0] - x) ** 2 + (q["pos"][1] - y) ** 2)[:8]
    los = [W.los(x, y, z + 6, q["pos"][0], q["pos"][1], q["pos"][2] + 6) for q in mechs]
    thr = mechs[0]["pos"] if mechs else None
    rays = []
    for k in range(16):
        ux, uy = math.cos(k * math.pi / 8), math.sin(k * math.pi / 8)
        rays.append((W.clear(x, y, z, ux, uy, 60.0, 1.5),
                     W.los(thr[0], thr[1], thr[2] + 6, x + ux * 20, y + uy * 20, z + 6) if thr else 1))
    f = lambda *s: rng.standard_normal(s).astype(np.float32)  # shapes only; values don't change cost
    return dict(self=f(95), mechs=f(8, 25), proj=f(8, 8), loot=f(8, 10), drops=f(3, 6), evt=f(30),
                rays=np.asarray(rays, np.float32).ravel(), offer=np.array([1, 5, 9])), los


class NetBot(Bot):
    def __init__(self, hello):
        super().__init__(hello)
        self.h = np.zeros(H, np.float32)
        self.t = {"brain": [], "features": [], "net": [], "total": []}
        inp, _ = features(self.world, {"self": {"pos": [0, 0, 0]}, "mechs": []})
        for _ in range(20):
            forward(inp, self.h)  # warm-up off the clock
        atexit.register(self.report)

    def act(self, obs):
        t0 = time.perf_counter(); a = super().act(obs)
        t1 = time.perf_counter(); inp, _ = features(self.world, obs)
        t2 = time.perf_counter(); _, self.h = forward(inp, self.h)
        t3 = time.perf_counter()
        for k, d in zip(self.t, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)):
            self.t[k].append(d * 1000)
        return a

    def report(self):
        print(f"NET_SCALE={S} params={N_PARAMS:,} ({N_PARAMS * 4 / 1e6:.1f} MB f32)", file=sys.stderr)
        for k, v in self.t.items():
            v = np.sort(v)
            if len(v):
                print(f"  {k:9s} ms: p50 {np.median(v):.2f}  p99 {v[int(.99 * len(v)) - 1]:.2f}  max {v[-1]:.2f}",
                      file=sys.stderr)


mf.run(NetBot)

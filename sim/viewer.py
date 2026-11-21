"""Record simulated matches and write self-contained HTML replay viewers.

  .venv/Scripts/python sim/viewer.py                      # practice: best genome vs 15 house bots
  .venv/Scripts/python sim/viewer.py --mode ffa           # best genome + 15 copies of --field
  .venv/Scripts/python sim/viewer.py --genome runs/main/best.json --seed 7 --out runs/viewer.html
  .venv/Scripts/python sim/viewer.py --seeds 20000-20019  # one replay per game + runs/replays/index.html
Genome default: newest runs/*/best.json, else bot/genome.json, else the defaults. --spec takes an ab.py
genome spec instead. Matches are deterministic and seated like house.practice / house.ffa, so a seed
replays the same game ab.py scored (ab.py: practice seed = seed0 + s, FFA seed = seed0 + 50000 + s).
"""
import argparse
import glob
import html as htmllib
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "bot")]
import params as PR  # noqa: E402
from ab import genome as spec_genome  # noqa: E402
from engine import DT, MAP_DIR, load_config  # noqa: E402
from evolve import _init  # noqa: E402
from house import ffa_match, practice_match  # noqa: E402

MODES = ["roam", "fight", "flee", "zone", "dodge"]
KINDS = {"gun": 0, "mortar": 1, "npc": 2}


def pick_genome(path):
    if path:
        return path, PR.load(path)
    runs = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "best.json")), key=os.path.getmtime)
    if runs:
        return runs[-1], PR.load(runs[-1])
    return "bot/genome.json or defaults", PR.load()


def build_match(cfg, mode, seed, genome, field):
    if mode == "practice":
        return practice_match(cfg, seed, genome)
    M, order = ffa_match(cfg, seed, [genome] + [field] * 15)
    return M, order.index(0), ["ours" if e == 0 else "rival" for e in order]


def intent(bot, act):
    """What our bot decided this tick: [goal or None, move x, move y, commanded yaw, pitch, action bits]."""
    g = getattr(bot, "goal", None)
    act = act if isinstance(act, dict) else {}
    mv, aim, f = act.get("move") or (0, 0), act.get("aim") or (0, 0), act.get("fire") or (0, 0)
    bits = (bool(f[0]) | bool(f[1]) << 1 | bool(act.get("boost")) << 2 | bool(act.get("jump")) << 3
            | (act.get("take") is not None) << 4)
    return [[round(g[0], 1), round(g[1], 1), round(g[2], 1), g[3]] if g else None,
            round(mv[0], 2), round(mv[1], 2), round(aim[0], 2), round(aim[1], 2), bits]


def record(M, every=4, slot=None, end_time=None):
    """Play M to the end. Returns frames, deaths, hits [t, attacker, target, hp lost, weapon] and
    shots (per mech, the times it fired). The engine is observed through wrappers, not changed."""
    frames, deaths, hits, beams = [], [], [], []
    shots = [[] for _ in M.mechs]
    last = {}
    if slot is not None:
        bot = M.mechs[slot].bot
        bot_act = bot.act

        def act(obs):
            last["act"] = bot_act(obs)
            return last["act"]
        bot.act = act
    damage, spawn = M._damage, M._spawn_proj

    def on_damage(tgt, amt, att, weapon, *a, **k):
        hp = tgt.hp
        damage(tgt, amt, att, weapon, *a, **k)
        if tgt.hp < hp and "mech" in (tgt.kind, getattr(att, "kind", None)):
            hits.append([round(M.t, 2), -1 if att is None else att.id, tgt.id, round(hp - tgt.hp), weapon])

    def on_spawn(kind, owner, *a, **k):
        if owner.kind == "mech":
            shots[owner.id].append(round(M.t, 2))
        return spawn(kind, owner, *a, **k)
    M._damage, M._spawn_proj = on_damage, on_spawn
    end = M.cfg["match"]["duration_s"] if end_time is None else end_time
    while M.t < end - 1e-9:
        if M.tick % 2 == 0:
            M.decide()
        t0 = M.t
        M.step()
        for b in M.beams:  # decide() clears M.beams every 2 ticks: keep each beam until the next frame
            if b[3] >= t0 - 1e-9:
                beams.append(b)
                shots[b[0]].append(round(b[3], 2))
        for m in M.mechs:
            if not m.alive and not any(d[1] == m.id for d in deaths):
                k = getattr(m, "killer", None)
                deaths.append([round(M.t, 1), m.id, m.place, k.id if k is not None and k.kind == "mech" else
                               (-2 if k is not None else -1)])
        if M.tick % every == 0 or (end_time is not None and (M.t >= end or sum(m.alive for m in M.mechs) <= 1)):
            zc, zr, ph = M.zone_at(M.t)
            MODES.extend({getattr(m.bot, "mode", "roam") for m in M.mechs} - set(MODES))  # brain.py adds modes
            frames.append({
                "t": round(M.t, 2),
                "m": [[round(m.x, 1), round(m.y, 1), round(m.yaw, 2), round(max(0.0, m.hp) / m.max_hp, 2),
                       1 if m.alive else 0, MODES.index(getattr(m.bot, "mode", "roam")),
                       "".join("t" if w["kind"] == "lightning" else w["kind"][0] for w in m.slots if w), m.cubes,
                       -1 if getattr(m.bot, "cur_target", None) is None else m.bot.cur_target,
                       round(m.z, 1), round(m.pitch, 2), int(m.jetting) | (not m.grounded) << 1] for m in M.mechs],
                "p": [[round(p.x, 1), round(p.y, 1), KINDS.get(p.kind, 0), round(p.z, 1), p.id] for p in M.projs],
                "b": [[0 if b[1] == "laser" else 1] + [round(v, 1) for pt in b[2] for v in pt] for b in beams],
                "z": [round(zc[0], 1), round(zc[1], 1), round(zr, 1), round(ph[4][0], 1), round(ph[4][1], 1), ph[5]],
                "n": [[round(n.x, 1), round(n.y, 1), round(n.z, 1), n.id] for n in M.npcs if n.alive],
                "d": [[round(d.x, 1), round(d.y, 1), 1 if d.landed else 0, round(d.z, 1)] for d in M.drops],
                "o": intent(M.mechs[slot].bot, last.get("act")) if slot is not None and M.mechs[slot].alive else None,
            })
            beams = []
        if sum(1 for m in M.mechs if m.alive) <= 1:
            break
    for k, m in enumerate(sorted((m for m in M.mechs if m.alive), key=lambda m: -m.hp)):
        m.place = k + 1
        deaths.append([round(M.t, 1), m.id, m.place, None])
    return frames, deaths, hits, shots


def static_map(map_dir=MAP_DIR):
    w = json.load(open(os.path.join(map_dir, "v2", "world.json")))
    prisms = [{"k": p["kind"][0], "h": round(p["top"] - p["base"], 1), "b": round(p["base"], 1),
               "r": [[round(v, 1) for v in ring] for ring in [p["ring"]] + list(p.get("holes") or [])]}
              for p in w["prisms"]]
    fi = {n: k for k, n in enumerate(w["trees"]["fields"])}
    n = len(fi)
    d = w["trees"]["data"]
    trees = [[round(d[i + fi[k]], 1) for k in ("crownX", "crownY", "crownR", "z", "crownBase", "height", "x", "y")]
             for i in range(0, len(d), n)]
    t = w["terrain"]  # every 4th vertex, in decimetres: enough for the 3D view, 16x smaller than the grid
    H = np.fromfile(os.path.join(map_dir, "v2", t["file"]), "<i2").reshape(t["ny"], t["nx"])[::4, ::4]
    terrain = {"x0": t["minX"], "y0": t["minY"], "step": t["step"] * 4, "nx": H.shape[1], "ny": H.shape[0],
               "h": np.round(H * t["scale"] * 10).astype(int).ravel().tolist()}
    barriers = [[b["height"]] + [round(v, 1) for v in b["line"]] for b in w["barriers"]]
    return {"bounds": w["bounds"], "prisms": prisms, "trees": trees, "terrain": terrain, "barriers": barriers}


def training_logs():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "log.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        out[os.path.basename(os.path.dirname(f))] = rows
    return out


def render(mode, seed, genome, field, gname, every, out):
    """Record one match into `out`. Returns [seed, place, t of our death or None, killer label, frames]."""
    M, slot, labels = build_match(load_config(), mode, seed, genome, field)
    frames, deaths, hits, shots = record(M, every, slot)
    mc = M.cfg["mech"]
    ours = next(d for d in deaths if d[1] == slot)
    data = {"mode": mode, "seed": seed, "genome": gname, "slot": slot, "labels": labels, "frames": frames,
            "deaths": deaths, "map": static_map(), "logs": training_logs(), "place": ours[2],
            "points": 16 - ours[2], "modes": MODES, "dt": every * DT, "hits": hits, "shots": shots,
            "mech": {k: mc[k] for k in ("radius", "height", "eye_height", "muzzle_height")},
            "dropR": M.cfg["drops"]["radius"]}
    html = open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8").read()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        three = open(os.path.join(HERE, "vendor", "three.min.js"), encoding="utf-8").read()  # inlined: one file
        f.write(html.replace("/*THREE*/", three).replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    k = ours[3]
    by = "survived" if k is None else "zone" if k == -1 else "drone" if k == -2 else f"{labels[k]} #{k}"
    return [seed, ours[2], None if k is None else ours[0], by, len(frames)]


def _render(args):
    return render(*args)


def parse_seeds(s):
    """'20000-20019' or '7,9,12-14' -> list of ints."""
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def write_index(out_dir, mode, gname, rows):
    rows = sorted(rows, key=lambda r: (-r[1], r[2] or 0))  # worst place first, then earliest death
    mean = sum(16 - r[1] for r in rows) / len(rows)
    body = "".join(
        f'<tr><td><a href="{mode}_{r[0]}.html">{r[0]}</a></td><td>{r[1]}</td><td>{16 - r[1]}</td>'
        f'<td>{"" if r[2] is None else f"{r[2]:.0f}s"}</td><td>{htmllib.escape(r[3])}</td></tr>' for r in rows)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Mech Replays</title>
<style>body{{background:#0e1116;color:#d7dde5;font:13px/1.5 system-ui,sans-serif;padding:16px}}
table{{border-collapse:collapse}}td,th{{padding:3px 14px;border-bottom:1px solid #2a313c;text-align:right}}
a{{color:#4aa3ff}}p{{color:#8a94a3}}</style>
<h1>Mech Royale replays</h1>
<p>{mode} · genome {htmllib.escape(gname)} · {len(rows)} games · mean {mean:.2f} pts · worst games first</p>
<table><tr><th>Seed</th><th>Place</th><th>Points</th><th>Died at</th><th>Killed by</th></tr>{body}</table></html>""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["practice", "ffa"], default="practice")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", help="batch: '20000-20019' or '7,9,12'; writes <mode>_<seed>.html + index.html")
    ap.add_argument("--genome", help="genome JSON file")
    ap.add_argument("--spec", help="ab.py genome spec (JSON) instead of --genome, e.g. '[\"off\", {\"sidestep\": 1}]'")
    ap.add_argument("--field", default='"off"', help="ab.py genome spec for the 15 FFA opponents")
    ap.add_argument("--every", type=int, default=4, help="record every Nth 0.05 s tick (1 = every state, 4x the file)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--out", help="output file, or directory with --seeds (default runs/viewer.html, runs/replays)")
    a = ap.parse_args()
    if a.spec:
        gname, genome = a.spec, spec_genome(json.loads(a.spec))
    else:
        gpath, genome = pick_genome(a.genome)
        gname = os.path.relpath(gpath, ROOT) if os.path.exists(gpath) else gpath
    field = spec_genome(json.loads(a.field))
    if not a.seeds:
        out = a.out or os.path.join(ROOT, "runs", "viewer.html")
        seed, place, _, by, n = render(a.mode, a.seed, genome, field, gname, a.every, out)
        print(f"wrote {out}: {a.mode} seed {seed}, our place {place} ({16 - place} pts, {by}), {n} frames")
        return
    out = a.out or os.path.join(ROOT, "runs", "replays")
    os.makedirs(out, exist_ok=True)
    tasks = [(a.mode, s, genome, field, gname, a.every, os.path.join(out, f"{a.mode}_{s}.html"))
             for s in parse_seeds(a.seeds)]
    with mp.Pool(min(a.workers, len(tasks)), initializer=_init) as pool:
        rows = pool.map(_render, tasks)
    write_index(out, a.mode, gname, rows)
    print(f"wrote {len(rows)} replays + {os.path.join(out, 'index.html')}: "
          f"mean {sum(16 - r[1] for r in rows) / len(rows):.2f} pts")


if __name__ == "__main__":
    main()

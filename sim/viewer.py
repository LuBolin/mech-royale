"""Record one simulated match and write a self-contained HTML viewer with training charts.

  .venv/Scripts/python sim/viewer.py                      # practice: best genome vs 15 house bots
  .venv/Scripts/python sim/viewer.py --mode ffa           # best genome + 15 hall-of-fame/default bots
  .venv/Scripts/python sim/viewer.py --genome runs/main/best.json --seed 7 --out runs/viewer.html
Genome default: newest runs/*/best.json, else bot/genome.json, else the defaults.
"""
import argparse
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "bot")]
import params as PR  # noqa: E402
from brain import Bot  # noqa: E402
from engine import MAP_DIR, Match, load_config  # noqa: E402
from house import ALL_HUNT_T, LEVELS, RANK, HouseBot  # noqa: E402

MODES = ["roam", "fight", "flee", "zone", "dodge"]
KINDS = {"gun": 0, "mortar": 1, "npc": 2}


def pick_genome(path):
    if path:
        return path, PR.load(path)
    runs = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "best.json")), key=os.path.getmtime)
    if runs:
        return runs[-1], PR.load(runs[-1])
    return "bot/genome.json or defaults", PR.load()


def build_match(cfg, mode, seed, genome):
    rng = random.Random(seed)
    slot = rng.randrange(16)
    labels, facs, teams, lv, shared = [], [], [], {}, None
    if mode == "practice":
        levels = LEVELS[:]
        rng.shuffle(levels)
        shared = {"player": slot, "last": None, "hunters": set(), "all_hunt_t": ALL_HUNT_T}
        for i in range(16):
            if i == slot:
                facs.append(lambda h: Bot(h, genome))
                teams.append(None)
                labels.append("ours")
            else:
                L = levels.pop()
                lv[i] = RANK[L]
                facs.append(lambda h, L=L: HouseBot(h, L, shared))
                teams.append(1)
                labels.append(L)
    else:
        hof = [PR.defaults()]
        for f in sorted(glob.glob(os.path.join(ROOT, "runs", "main", "mean_g*.json")))[-15:]:
            hof.append(PR.load(f))
        for i in range(16):
            g = genome if i == slot else rng.choice(hof)
            facs.append(lambda h, g=g: Bot(h, g))
            teams.append(None)
            labels.append("ours" if i == slot else "rival")
    M = Match(cfg, seed, facs, teams or None, shared=shared, levels=lv)
    return M, slot, labels


def record(M, every=4):
    frames, deaths = [], []
    end = M.cfg["match"]["duration_s"]
    while M.t < end - 1e-9:
        if M.tick % 2 == 0:
            M.decide()
        M.step()
        for m in M.mechs:
            if not m.alive and not any(d[1] == m.id for d in deaths):
                k = getattr(m, "killer", None)
                deaths.append([round(M.t, 1), m.id, m.place, k.id if k is not None and k.kind == "mech" else
                               (-2 if k is not None else -1)])
        if M.tick % every == 0:
            zc, zr, ph = M.zone_at(M.t)
            frames.append({
                "t": round(M.t, 1),
                "m": [[round(m.x, 1), round(m.y, 1), round(m.yaw, 2), round(max(0.0, m.hp) / m.max_hp, 2),
                       1 if m.alive else 0, MODES.index(getattr(m.bot, "mode", "roam")) if hasattr(m.bot, "mode") else 0,
                       "".join(w["kind"][0] for w in m.slots if w), m.cubes] for m in M.mechs],
                "p": [[round(p.x, 1), round(p.y, 1), KINDS.get(p.kind, 0)] for p in M.projs],
                "b": [[round(b[2][0][0], 1), round(b[2][0][1], 1), round(b[2][-1][0], 1), round(b[2][-1][1], 1),
                       0 if b[1] == "laser" else 1] for b in M.beams],
                "z": [round(zc[0], 1), round(zc[1], 1), round(zr, 1), round(ph[4][0], 1), round(ph[4][1], 1), ph[5]],
                "n": [[round(n.x, 1), round(n.y, 1)] for n in M.npcs if n.alive],
                "d": [[round(d.x, 1), round(d.y, 1), 1 if d.landed else 0] for d in M.drops],
            })
        if sum(1 for m in M.mechs if m.alive) <= 1:
            break
    for k, m in enumerate(sorted((m for m in M.mechs if m.alive), key=lambda m: -m.hp)):
        m.place = k + 1
        deaths.append([round(M.t, 1), m.id, m.place, None])
    return frames, deaths


def static_map():
    w = json.load(open(os.path.join(MAP_DIR, "v2", "world.json")))
    prisms = [{"k": p["kind"][0], "h": round(p["top"] - p["base"], 1),
               "r": [[round(v, 1) for v in ring] for ring in [p["ring"]] + list(p.get("holes") or [])]}
              for p in w["prisms"]]
    fi = {n: k for k, n in enumerate(w["trees"]["fields"])}
    n = len(fi)
    d = w["trees"]["data"]
    trees = [[round(d[i + fi["x"]], 1), round(d[i + fi["y"]], 1), round(d[i + fi["crownR"]], 1)]
             for i in range(0, len(d), n)]
    return {"bounds": w["bounds"], "prisms": prisms, "trees": trees}


def training_logs():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "log.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        out[os.path.basename(os.path.dirname(f))] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["practice", "ffa"], default="practice")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--genome")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "viewer.html"))
    a = ap.parse_args()
    cfg = load_config()
    gpath, genome = pick_genome(a.genome)
    M, slot, labels = build_match(cfg, a.mode, a.seed, genome)
    frames, deaths = record(M)
    ours = next(d for d in deaths if d[1] == slot)
    data = {"mode": a.mode, "seed": a.seed, "genome": os.path.relpath(gpath, ROOT) if os.path.exists(gpath) else gpath,
            "slot": slot, "labels": labels, "frames": frames, "deaths": deaths, "map": static_map(),
            "logs": training_logs(), "place": ours[2], "points": 16 - ours[2], "modes": MODES}
    html = open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8").read()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {a.out}: {a.mode} seed {a.seed}, our place {ours[2]} ({16 - ours[2]} pts), {len(frames)} frames")


if __name__ == "__main__":
    main()

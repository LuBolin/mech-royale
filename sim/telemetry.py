"""Engine-side telemetry of the player's mech in practice matches, identical for in-process and Docker bots.

An entry is LABEL=ours:DIR (in-process bot from DIR) or a roster name (sandboxed DockerBot via sim/league.py).

    python sim/telemetry.py --entry base=ours:runs/variants/base/bot --entry combo=ours:runs/variants/combo/bot \
        --entry giddy --entry azing --entry kanya --seeds 40 --seed-base 6000 --jobs 6
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import statistics as st
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import league  # noqa: E402  (DockerBot, load, factory, freeze_house; never modified here)

OUT = league.OUT / "telemetry.csv"
DT = 0.05
EXPOSE_R = 200.0
PHASES = [(0, 30), (30, 90), (90, 180), (180, 1e9)]
KINDS = ["gun", "mortar", "laser", "lightning"]


def quart(xs):
    if not xs:
        return ["", "", ""]
    q = np.percentile(xs, [25, 50, 75])
    return [round(float(v), 1) for v in q]


def play(task):
    label, entry, seed, images = task
    t0 = time.time()
    try:
        M, slot, _ = league.house.practice_match(league.CFG, seed, None, player=league.factory(entry, images))
        me = M.mechs[slot]
        walk = me.stats["walk_speed"]
        R = defaultdict(float)  # accumulators
        first = {}
        dmg_dist_dealt, dmg_dist_taken = [], []
        taken_w, dealt_w, shots, hits = defaultdict(float), defaultdict(float), defaultdict(int), defaultdict(int)
        exp_dmg, exp_time = defaultdict(float), defaultdict(float)
        ph_dmg, ph_time = defaultdict(float), defaultdict(float)
        hold = defaultdict(float)
        st_ = dict(expo=0, last_hit_t=-99.0, prev_head=None, shot=None, shot_hit=False)

        def mark(k, v=True):
            if k not in first:
                first[k] = (round(M.t, 2), v)

        def phase(t):
            return next(i for i, (a, b) in enumerate(PHASES) if a <= t < b)

        damage = M._damage

        def tracked_damage(tgt, amt, att, weapon, src, **k):
            hp, alive = max(tgt.hp, 0.0), tgt.alive
            damage(tgt, amt, att, weapon, src, **k)
            lost = hp - max(tgt.hp, 0.0)
            if tgt is me and lost > 0 and att is not me:
                taken_w[weapon] += lost
                exp_dmg[min(st_["expo"], 3)] += lost
                ph_dmg[phase(M.t)] += lost
                st_["last_hit_t"] = M.t
                mark("first_taken")
                if att is not None and att.kind == "mech":
                    dmg_dist_taken.append(math.hypot(att.x - me.x, att.y - me.y))
            if att is me and tgt is not me:
                if tgt.kind == "mech" and lost > 0:
                    dealt_w[weapon] += lost
                    dmg_dist_dealt.append(math.hypot(tgt.x - me.x, tgt.y - me.y))
                    mark("first_dealt")
                    if weapon == "gun":
                        hits["gun"] += 1
                    else:
                        st_["shot_hit"] = True
                if alive and not tgt.alive:
                    R["crates" if tgt.kind == "crate" else "drones"] += tgt.kind in ("crate", "npc")
        M._damage = tracked_damage

        spawn = M._spawn_proj

        def tracked_spawn(kind, owner, *a, **k):
            if owner is me and kind in ("gun", "mortar"):
                shots[kind] += 1
            return spawn(kind, owner, *a, **k)
        M._spawn_proj = tracked_spawn

        def shot_wrap(name, kind):  # one hit per mortar explosion / laser beam / lightning cast that hurt a mech
            fn = getattr(M, name)

            def wrapped(p_or_m, *a, **k):
                mine = (p_or_m.owner if name == "_explode" else p_or_m) is me
                st_["shot_hit"] = False
                fn(p_or_m, *a, **k)
                if mine:
                    if name != "_explode":
                        shots[kind] += 1
                    hits[kind] += st_["shot_hit"]
            setattr(M, name, wrapped)
        for name, kind in (("_explode", "mortar"), ("_laser", "laser"), ("_lightning", "lightning")):
            shot_wrap(name, kind)

        step = M.step

        def sample_exposure():
            foes = [m for m in M.mechs if m.alive and m is not me and math.hypot(m.x - me.x, m.y - me.y) <= EXPOSE_R]
            near = min((math.hypot(m.x - me.x, m.y - me.y) for m in M.mechs if m.alive and m is not me), default=None)
            if near is not None:
                R["near_sum"] += near
                R["near_n"] += 1
            if not foes:
                return 0
            A = np.array([(m.x, m.y, m.z + m.stats["muzzle_height"]) for m in foes], float)
            B = np.tile((me.x, me.y, me.z + me.h / 2), (len(foes), 1))
            return int((M.G.ray_many(A, B, False) > 1.0).sum())

        def tracked_step():
            if not me.alive:
                return step()
            t = M.t
            if M.tick % 2 == 0:
                st_["expo"] = sample_exposure()
                if st_["expo"]:
                    mark("first_exposed")
                if M.shared and M.shared.get("last"):
                    mark("first_spotted")
                hunters = (M.shared or {}).get("hunters", ())
                if any(M.mechs[h].alive and math.hypot(M.mechs[h].x - me.x, M.mechs[h].y - me.y) <= EXPOSE_R
                       for h in hunters):
                    mark("first_hunter_near")
            mv = me.move
            boosting = me.boost and me.grounded and not me.jump and mv[0] ** 2 + mv[1] ** 2 > 1e-4 and me.energy > 0
            x0, y0 = me.x, me.y
            step()
            d = math.hypot(me.x - x0, me.y - y0)
            spd = d / DT
            R["alive"] += DT
            R["dist"] += d
            R["fast"] += DT * (spd > 0.8 * walk)
            R["still"] += DT * (spd < 1.0)
            R["boost"] += DT * boosting
            R["jet"] += DT * me.jetting
            R["air"] += DT * (not me.grounded)
            e = min(st_["expo"], 3)
            exp_time[e] += DT
            R["expo_sum"] += st_["expo"] * DT
            ph_time[phase(t)] += DT
            if not me.in_zone:
                R["out_zone"] += DT
                R["zone_dmg"] += M.zone_at(M.t)[2][6] * DT
            kinds = me.kinds()
            for k in kinds:
                hold[k] += DT
            if len(kinds) == 2:
                mark("two_weapons")
            if any(k != "gun" for k in kinds):
                mark("first_weapon", next(k for k in kinds if k != "gun"))
            if me.cubes:
                mark("first_cube")
            if spd > 1.0:
                head = math.atan2(me.y - y0, me.x - x0)
                if st_["prev_head"] is not None and M.t - st_["last_hit_t"] < 1.0:
                    R["jit_sum"] += abs(math.remainder(head - st_["prev_head"], 2 * math.pi))
                    R["jit_t"] += DT
                st_["prev_head"] = head
            else:
                st_["prev_head"] = None
        M.step = tracked_step
        res = M.run()
    finally:
        league._close_all()
    alive = max(R["alive"], DT)
    cause = ""
    if not me.alive:
        k = getattr(me, "killer", None)
        cause = "zone" if k is None else "self" if k is me else k.kind if k.kind != "mech" else \
            max(taken_w, key=taken_w.get, default="mech")
    pts = 16 - res["place"][slot] if res["place"][slot] else 0
    row = dict(entry=label, seed=seed, points=pts, kills=me.kills, survived_s=round(alive, 2), cause=cause,
               t_first_weapon=first.get("first_weapon", ("",))[0], first_weapon=first.get("first_weapon", ("", ""))[1],
               t_first_cube=first.get("first_cube", ("",))[0], t_first_taken=first.get("first_taken", ("",))[0],
               t_first_dealt=first.get("first_dealt", ("",))[0], t_first_exposed=first.get("first_exposed", ("",))[0],
               t_first_spotted=first.get("first_spotted", ("",))[0],
               t_first_hunter_near=first.get("first_hunter_near", ("",))[0],
               t_two_weapons=first.get("two_weapons", ("",))[0],
               m_per_s=round(R["dist"] / alive, 2), fast_share=round(R["fast"] / alive, 3),
               boost_share=round(R["boost"] / alive, 3), jet_share=round(R["jet"] / alive, 3),
               air_share=round(R["air"] / alive, 3), still_share=round(R["still"] / alive, 3),
               jitter_rad_s=round(R["jit_sum"] / R["jit_t"], 2) if R["jit_t"] else "",
               expo_mean=round(R["expo_sum"] / alive, 3),
               near_enemy_m=round(R["near_sum"] / R["near_n"], 1) if R["near_n"] else "",
               taken=round(sum(taken_w.values()), 1), dealt=round(sum(dealt_w.values()), 1),
               taken_ps=round(sum(taken_w.values()) / alive, 2), dealt_ps=round(sum(dealt_w.values()) / alive, 2),
               out_zone_s=round(R["out_zone"], 2), zone_dmg=round(R["zone_dmg"], 1),
               cubes=me.cubes, upgrades="|".join(me.upgrades), crates=int(R["crates"]), drones=int(R["drones"]),
               max_hp=round(me.max_hp, 1), match_s=round(time.time() - t0, 2))
    for e in range(4):
        row[f"expo{e}_share"] = round(exp_time[e] / alive, 3)
        row[f"taken_ps_expo{e}"] = round(exp_dmg[e] / exp_time[e], 2) if exp_time[e] > 1 else ""
    for i, (a, b) in enumerate(PHASES):
        tag = f"{a}_{b if b < 1e8 else 'end'}"
        row[f"alive_{tag}"] = round(ph_time[i], 2)
        row[f"taken_ps_{tag}"] = round(ph_dmg[i] / ph_time[i], 2) if ph_time[i] > 1 else ""
    for k in KINDS:
        row[f"hold_{k}"] = round(hold[k] / alive, 3)
        row[f"taken_{k}"] = round(taken_w[k], 1)
        row[f"dealt_{k}"] = round(dealt_w[k], 1)
        row[f"shots_{k}"] = shots[k]
        row[f"hits_{k}"] = hits[k]
    row["taken_npc"] = round(taken_w["npc"], 1)
    row["dist_dealt_q"] = "|".join(map(str, quart(dmg_dist_dealt)))
    row["dist_dealt_mean"] = round(st.mean(dmg_dist_dealt), 1) if dmg_dist_dealt else ""
    row["dist_taken_q"] = "|".join(map(str, quart(dmg_dist_taken)))
    row["dist_taken_mean"] = round(st.mean(dmg_dist_taken), 1) if dmg_dist_taken else ""
    return row


def table(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["entry"]].append(r)
    names = list(dict.fromkeys(r["entry"] for r in rows))
    num = lambda v: v not in ("", None) and not isinstance(v, str) or (isinstance(v, str) and v.replace(".", "", 1).replace("-", "", 1).isdigit())
    cols = [c for c in rows[0] if c not in ("entry", "seed", "match_s") and any(num(r[c]) for r in rows)]
    print(f"{'metric':<22}" + "".join(f"{n:>10}" for n in names))
    for c in cols:
        cells = []
        for n in names:
            xs = [float(r[c]) for r in by[n] if num(r[c])]
            cells.append(f"{st.mean(xs):>10.3g}" if xs else f"{'-':>10}")
        print(f"{c:<22}" + "".join(cells))
    for n in names:  # derived rows that are ratios of sums, not means of ratios
        rs = by[n]
        hr = {k: (sum(r[f"hits_{k}"] for r in rs), sum(r[f"shots_{k}"] for r in rs)) for k in KINDS}
        print(n, "hit rate", {k: f"{h / s:.2f}({s})" for k, (h, s) in hr.items() if s},
              "median surv", st.median(r["survived_s"] for r in rs),
              "first weapon", dict(sorted(defaultdict(int, {w: sum(r["first_weapon"] == w for r in rs) for w in KINDS[1:]}).items())),
              "causes", {c: sum(r["cause"] == c for r in rs) for c in set(r["cause"] for r in rs)})


def parse(spec):
    label, _, rest = spec.partition("=")
    if rest.startswith("ours:"):
        return label, "ours", Path(rest[5:]).resolve()
    return spec, spec, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", action="append", required=True)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--seed-base", type=int, default=6000)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--table", action="store_true", help="only print the table of an existing --out")
    args = ap.parse_args()
    if args.table:
        with args.out.open() as f:
            rows = [{k: (float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v) for k, v in r.items()}
                    for r in csv.DictReader(f)]
        return table(rows)
    entries = [parse(e) for e in args.entry]
    roster = json.loads(league.ROSTER.read_text())
    images = {e: roster.get(e, {}).get("image", f"league-{e}") for _, e, _ in entries if e != "ours"}
    for n, img in images.items():
        if subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode:
            sys.exit(f"entry {n}: docker image {img} not found")
    league.freeze_house()
    os.environ["LEAGUE_RUN"] = "tel" + uuid.uuid4().hex[:6]
    seeds = range(args.seed_base, args.seed_base + args.seeds)
    # league.load binds one `ours` per process, so each ours dir gets its own pool; Docker entries ride the first.
    dirs = list(dict.fromkeys(d for _, _, d in entries if d)) or [Path(league.ROOT / "bot")]
    groups = {d: [] for d in dirs}
    for label, e, d in entries:
        groups[d or dirs[0]] += [(label, e, s, images) for s in seeds]
    rows, t0 = [], time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    f = w = None
    try:
        for d, tasks in groups.items():
            pool = mp.get_context("spawn").Pool(args.jobs, initializer=league.load, initargs=(d,), maxtasksperchild=50)
            try:
                for r in pool.imap_unordered(play, tasks):
                    if w is None:
                        f = args.out.open("w", newline="")
                        w = csv.DictWriter(f, list(r))
                        w.writeheader()
                    w.writerow(r)
                    f.flush()
                    rows.append(r)
                    print(f"{r['entry']} {r['seed']}: {r['points']} pts, {r['survived_s']} s ({r['match_s']} s)", flush=True)
                pool.close()
                pool.join()
            finally:
                pool.terminate()
    finally:
        if f:
            f.close()
        ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=league-{os.environ['LEAGUE_RUN']}-"],
                             capture_output=True, text=True).stdout.split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
    print(f"{len(rows)} matches in {time.time() - t0:.0f} s; rows in {args.out}")
    table(rows)


if __name__ == "__main__":
    main()

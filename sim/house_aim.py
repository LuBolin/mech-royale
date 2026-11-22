"""House gun hit rate on the player by range, in practice matches (to calibrate house aim against real data).

    python sim/house_aim.py --house v2 --seeds 20 [--ours bot]

Real house hit rates against top bots (docs/competitor-survey.md): about 88% under 40 m, 55% at 40-80 m,
30% at 80-120 m, 15% at 120-180 m, 3% beyond.
"""
import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BINS = (40, 80, 120, 180, 1e9)
REAL = (0.88, 0.55, 0.30, 0.15, 0.03)


def measure(seed, house):
    import league
    M, slot, _ = league.house.practice_match(league.CFG, seed, None, player=league.factory("ours", {}), house=house)
    me = M.mechs[slot]
    flying = defaultdict(list)  # house shooter id -> range bins of its in-flight rounds aimed at us, oldest first
    shots, hits = defaultdict(int), defaultdict(int)
    spawn, damage, step_proj = M._spawn_proj, M._damage, M._projectiles

    def tracked_spawn(kind, owner, dx, dy, dz, *a, **k):
        p = spawn(kind, owner, dx, dy, dz, *a, **k)
        if kind == "gun" and owner is not me and me.alive:
            d = math.hypot(me.x - owner.x, me.y - owner.y)
            off = abs((math.atan2(dy, dx) - math.atan2(me.y - owner.y, me.x - owner.x) + math.pi) % (2 * math.pi) - math.pi)
            if off < max(0.1, math.atan2(12, d)):  # aimed at us, not at a drone or crate
                b = next(i for i, e in enumerate(BINS) if d < e)
                shots[b] += 1
                p.calib_bin = b
                flying[owner.id].append(p)
        return p

    def tracked_damage(tgt, amt, att, weapon, src, *a, **k):
        if tgt is me and weapon == "gun" and att is not None and flying.get(att.id):
            hits[flying[att.id].pop(0).calib_bin] += 1
        return damage(tgt, amt, att, weapon, src, *a, **k)

    def tracked_projectiles():
        step_proj()
        live = {id(p) for p in M.projs}
        for k in flying:
            flying[k] = [p for p in flying[k] if id(p) in live]

    M._spawn_proj, M._damage, M._projectiles = tracked_spawn, tracked_damage, tracked_projectiles
    M.run()
    return shots, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", default="v2")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--ours", type=Path, default=HERE.parent / "bot")
    args = ap.parse_args()
    os.environ["LEAGUE_HOUSE"] = args.house
    import league
    league.load(args.ours)
    S, H = defaultdict(int), defaultdict(int)
    for s in range(args.seed_base, args.seed_base + args.seeds):
        shots, hits = measure(s, args.house)
        for b in shots:
            S[b] += shots[b]
            H[b] += hits[b]
    lo = 0
    for b, (e, real) in enumerate(zip(BINS, REAL)):
        print(f"{lo:>4}-{e if e < 1e9 else 'inf':<4} m  shots {S[b]:5d}  hit {H[b] / max(1, S[b]):.2f}  real {real:.2f}")
        lo = e


if __name__ == "__main__":
    main()

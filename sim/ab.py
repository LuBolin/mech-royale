"""Paired A/B of genome variants in the simulator.

Every variant plays the same seeds as the base: practice (vs 15 house bots) and FFA (one seat among 15
copies of the field genome). Prints mean placement points and the paired difference with its standard error.

  python sim/ab.py --base off --variants '{"sidestep": {"sidestep": 1}}' --prac 300 --ffa 100

"off" = defaults with params.OFF (the pre-survey bot); "def" = current defaults. Overrides apply on top.
The sim mis-ranks tactics against the real board (docs/competitor-survey.md): use it to see that a feature
fires and does no harm, not as the final word.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bot"))
import params as PR  # noqa: E402
from evolve import _init, _task  # noqa: E402


def genome(spec):
    """'off' | 'def' | dict of overrides on defaults | ['off', {...}]."""
    if isinstance(spec, list):
        g = genome(spec[0])
        g.update(spec[1])
        return g
    if spec == "off":
        g = PR.defaults()
        g.update(PR.OFF)
        return g
    g = PR.defaults()
    if isinstance(spec, dict):
        g.update(spec)
    return g


def stats(a, b):
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / max(1, n - 1))
    return sum(b) / n, m, sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default='"off"', help="JSON genome spec for the base")
    ap.add_argument("--field", default='"off"', help="JSON genome spec for the 15 FFA opponents")
    ap.add_argument("--variants", required=True, help="JSON {name: overrides applied on the base}")
    ap.add_argument("--prac", type=int, default=300)
    ap.add_argument("--ffa", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = ap.parse_args()
    base = genome(json.loads(args.base))
    field = genome(json.loads(args.field))
    variants = {"base": base}
    for name, ov in json.loads(args.variants).items():
        g = dict(base)
        g.update(ov)
        variants[name] = g
    names = list(variants)
    tasks = []
    for i, n in enumerate(names):
        tasks += [("prac", (i, s), args.seed0 + s, variants[n]) for s in range(args.prac)]
        tasks += [("ffa", (i, s), args.seed0 + 50000 + s, [variants[n]] + [field] * 15) for s in range(args.ffa)]
    prac = {n: [0.0] * args.prac for n in names}
    ffa = {n: [0.0] * args.ffa for n in names}
    with mp.Pool(args.workers, initializer=_init) as pool:
        for kind, (i, s), res in pool.imap_unordered(_task, tasks, chunksize=2):
            if kind == "prac":
                prac[names[i]][s] = res
            else:
                ffa[names[i]][s] = res[0]
    print(f"{'variant':<16}{'practice':>10}{'d_prac':>9}{'se':>7}{'ffa':>9}{'d_ffa':>9}{'se':>7}")
    out = {}
    for n in names:
        pm, pd, pse = stats(prac["base"], prac[n])
        fm, fd, fse = stats(ffa["base"], ffa[n]) if args.ffa else (0, 0, 0)
        out[n] = dict(prac=pm, d_prac=pd, se_prac=pse, ffa=fm, d_ffa=fd, se_ffa=fse)
        print(f"{n:<16}{pm:>10.3f}{pd:>+9.3f}{pse:>7.3f}{fm:>9.3f}{fd:>+9.3f}{fse:>7.3f}", flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

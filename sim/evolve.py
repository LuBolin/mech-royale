"""CMA-ES over the bot genome, scored in the headless simulator.

One generation (= one iteration):
  * free-for-alls: each match seats 16 - --hof candidates plus --hof opponents drawn from a hall of
    fame (the default genome and past benchmark means), so candidates don't only learn to beat each
    other; matches are scheduled so every candidate plays about --ffa of them;
  * --prac practice matches per candidate (shared seeds) against 15 house bots;
  * fitness = w * mean FFA points + (1 - w) * mean practice points (placement points, 0..15).
Every --bench generations the distribution mean is scored on fixed seeds: practice, and FFA as
1 mean genome vs 15 default genomes. That score is comparable across runs.

  python sim/evolve.py --out runs/main --gens 2000
  python sim/evolve.py --out runs/main --gens 2000 --resume
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bot"))
import params as PR  # noqa: E402
from cmaes import CMA  # noqa: E402

_CFG = None


def _init():
    global _CFG
    from engine import load_config
    from world import get_world
    from engine import MAP_DIR
    _CFG = load_config()
    get_world(MAP_DIR)


def _task(task):
    from house import ffa, practice
    kind, key, seed, payload = task
    if kind == "ffa":
        pts, _ = ffa(_CFG, seed, payload)
        return kind, key, pts
    p, _ = practice(_CFG, seed, payload)
    return kind, key, p


def evaluate(pool, X, args, rng, hof):
    """X: (lam, n) in [0,1]-space. Returns fitness, ffa means, practice means."""
    lam = len(X)
    P = [PR.decode(x) for x in X]
    tasks = []
    seats = 16 - args.hof
    plays = [0] * lam
    for _ in range(math.ceil(args.ffa * lam / seats)):
        grp = sorted(range(lam), key=lambda i: (plays[i], rng.random()))[:seats]
        for i in grp:
            plays[i] += 1
        opp = [rng.choice(hof) for _ in range(args.hof)]
        tasks.append(("ffa", grp, rng.randrange(2 ** 31), [P[i] for i in grp] + opp))
    pseeds = [rng.randrange(2 ** 31) for _ in range(args.prac)]
    for i in range(lam):
        for s in pseeds:
            tasks.append(("prac", i, s, P[i]))
    ffa_pts = [[] for _ in range(lam)]
    prac_pts = [[] for _ in range(lam)]
    for kind, key, res in pool.imap_unordered(_task, tasks, chunksize=1):
        if kind == "ffa":
            for i, v in zip(key, res):
                ffa_pts[i].append(v)
        else:
            prac_pts[key].append(res)
    fm = np.array([np.mean(v) if v else 0.0 for v in ffa_pts])
    pm = np.array([np.mean(v) if v else 0.0 for v in prac_pts])
    return args.w_ffa * fm + (1 - args.w_ffa) * pm, fm, pm


def benchmark(pool, x, n_prac=32, n_ffa=16):
    """Fixed-seed score of genome x: (practice mean, FFA mean vs 15 default genomes)."""
    p = PR.decode(x)
    d = PR.defaults()
    tasks = [("prac", 0, 1000 + s, p) for s in range(n_prac)]
    tasks += [("ffa", 0, 5000 + s, [p] + [d] * 15) for s in range(n_ffa)]
    pr, ff = [], []
    for kind, _, res in pool.imap_unordered(_task, tasks, chunksize=1):
        (ff.append(res[0]) if kind == "ffa" else pr.append(res))
    return float(np.mean(pr)), float(np.mean(ff))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--gens", type=int, default=2000)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--ffa", type=int, default=12, help="FFA rounds per generation")
    ap.add_argument("--prac", type=int, default=2, help="practice matches per candidate per generation")
    ap.add_argument("--w_ffa", type=float, default=0.5)
    ap.add_argument("--hof", type=int, default=4, help="hall-of-fame opponents per FFA match")
    ap.add_argument("--bench", type=int, default=25, help="benchmark the mean every N generations")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 8))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    assert 0 <= args.hof < 16
    os.makedirs(args.out, exist_ok=True)
    ck = os.path.join(args.out, "state.pkl")
    if args.resume and os.path.exists(ck):
        with open(ck, "rb") as f:
            es, rng, gen, best, hof = pickle.load(f)
    else:
        es = CMA(PR.encode(PR.defaults()), args.sigma, args.pop, args.seed)
        rng, gen, best, hof = random.Random(args.seed), 0, (-1.0, None), [PR.defaults()]
        with open(os.path.join(args.out, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=1)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    with mp.Pool(args.workers, initializer=_init) as pool:
        if gen == 0:
            b = benchmark(pool, es.mean)
            log.write(json.dumps({"gen": 0, "bench_prac": b[0], "bench_ffa": b[1]}) + "\n")
            log.flush()
            print("gen 0 bench", b, flush=True)
        while gen < args.gens:
            t0 = time.time()
            X = es.ask()
            Xc = np.clip(X, 0, 1)
            fit, fm, pm = evaluate(pool, Xc, args, rng, hof)
            pen = 10 * ((X - Xc) ** 2).sum(1)
            es.tell(X, -(fit - pen))
            gen += 1
            rec = {"gen": gen, "fit_best": float(fit.max()), "fit_mean": float(fit.mean()),
                   "prac_mean": float(pm.mean()), "prac_best": float(pm.max()), "sigma": es.sigma,
                   "sec": round(time.time() - t0, 1)}
            if gen % args.bench == 0 or gen == args.gens:
                b = benchmark(pool, np.clip(es.mean, 0, 1))
                rec.update(bench_prac=b[0], bench_ffa=b[1])
                score = b[0] + b[1]
                genome = PR.decode(np.clip(es.mean, 0, 1))
                hof = (hof + [genome])[-20:]
                with open(os.path.join(args.out, f"mean_g{gen:04d}.json"), "w") as f:
                    json.dump(genome, f, indent=1)
                if score > best[0]:
                    best = (score, gen)
                    with open(os.path.join(args.out, "best.json"), "w") as f:
                        json.dump(genome, f, indent=1)
                rec["best_gen"] = best[1]
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            with open(ck + ".tmp", "wb") as f:
                pickle.dump((es, rng, gen, best, hof), f)
            os.replace(ck + ".tmp", ck)


if __name__ == "__main__":
    main()

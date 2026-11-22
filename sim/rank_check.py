"""How well a house model ranks bots with known board scores. Usage: rank_check.py <house label> ...

Reads runs/league/rank-<label>-dock-practice.csv (ours = commit 35f4568, plus competitor images) and
rank-<label>-{b396,b425,every}-practice.csv (our other commits / the current bot, in process).
"""
import csv
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOARD = {"giddy": 5.96, "kanya": 5.83, "azing": 5.96, "mike": 5.52, "anthony": 5.13, "whymiso": 4.56,
         "overcooked": 3.38, "onepizza": 2.5, "fraun": 1.58, "b473": 4.73, "b396": 3.96, "b425": 4.25}


def rows(name):
    path = ROOT / "runs" / "league" / f"{name}-practice.csv"
    return list(csv.DictReader(path.open())) if path.exists() else []


def main():
    for h in sys.argv[1:]:
        sim = {}
        for r in rows(f"rank-{h}-dock"):
            sim.setdefault("b473" if r["entry"] == "ours" else r["entry"], []).append(float(r["points"]))
        for v in ("b396", "b425", "every"):
            sim[v] = [float(r["points"]) for r in rows(f"rank-{h}-{v}")]
        keys = [k for k in BOARD if sim.get(k)]
        xs, ys = [BOARD[k] for k in keys], [S.mean(sim[k]) for k in keys]
        rank = S.correlation([sorted(xs).index(x) for x in xs], [sorted(ys).index(y) for y in ys])
        fit = S.linear_regression(ys, xs)  # board ~ a * sim + b
        resid = {k: BOARD[k] - (fit.slope * S.mean(sim[k]) + fit.intercept) for k in keys}
        print(f"== {h}: pearson {S.correlation(xs, ys):.2f}  spearman {rank:.2f}  "
              f"rms miss {S.fmean(v * v for v in resid.values()) ** 0.5:.2f}")
        for k in sorted((k for k in sim if sim[k]), key=lambda k: -S.mean(sim[k])):
            b = BOARD.get(k)
            print(f"  {k:11} n={len(sim[k]):3} sim {S.mean(sim[k]):5.2f}  board {b if b else '':>5}"
                  + (f"  miss {resid[k]:+.2f}" if k in resid else ""))


if __name__ == "__main__":
    main()

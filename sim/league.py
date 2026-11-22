"""League: our bot against frozen house bots and sandboxed competitor images.

practice: each --entry plays alone against 15 house bots on the same seeds. The house bots run the
    bot/ of commit HOUSE_REV (extracted to runs/league/house/bot), so they do not change with ours.
ffa: --ours-seats copies of ours plus a --field of competitor images, seats shuffled by seed.

An entry is `ours` (in process, from --ours DIR) or a name from sim/league_roster.json
({name: {"image": ...}}; a name missing from it runs image league-<name>). Competitor code only ever
runs inside `docker run --network none --read-only`, one container per seat.

    python sim/league.py practice --entry ours --entry giddy --seeds 24 --jobs 8 --label base
    python sim/league.py practice --entry ours --house v2 --seeds 32 --label v2   # evidence-based house
    python sim/league.py ffa --field giddy:4,kanya:4,azing:4,mike:3 --seeds 16 --jobs 6 --label base
"""
import argparse
import atexit
import csv
import io
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tarfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from queue import Queue, Empty
from threading import Thread

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "runs" / "league"
HOUSE_REV = "0ee12ad"
HOUSE_DIR = OUT / "house" / "bot"
ARENA = ROOT / "kit" / "arena"
ROSTER = HERE / "league_roster.json"
READY_S, ANSWER_S, LATE_S = 20.0, 1.0, 0.05
FIELDS = ["mode", "seed", "entry", "seat", "points", "place", "kills", "dealt", "taken", "survived_s", "cause",
          "late", "missed", "answer_ms", "match_s"]
LIVE = set()


class DockerBot:
    """One competitor container. Never raises into the match: a slow, dead or garbled bot answers None."""

    def __init__(self, image, hello):
        self.name = f"league-{os.environ.get('LEAGUE_RUN', 'x')}-{uuid.uuid4().hex[:8]}"
        self.late = self.missed = self.answers = 0
        self.wait_s, self.sent, self.ready = 0.0, None, None
        LIVE.add(self)
        self.proc = subprocess.Popen(
            ["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--tmpfs", "/tmp", "--cpus", "1",
             "--memory", "512m", "--name", self.name, "-v", f"{ARENA}:/arena:ro", image],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        self.lines, self.out = Queue(), Queue()
        Thread(target=self._read, daemon=True).start()
        Thread(target=self._write, daemon=True).start()
        self.out.put(dict(hello, map=dict(hello["map"], dir="/arena")))
        self.ready_by = time.monotonic() + READY_S  # containers start in parallel; first send waits

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put((time.monotonic(), line))
        self.lines.put((time.monotonic(), None))

    def _write(self):
        try:
            for msg in iter(self.out.get, None):
                self.proc.stdin.write(json.dumps(msg, default=float) + "\n")
                self.proc.stdin.flush()
        except (OSError, ValueError):
            pass  # the bot died; its reader sees EOF

    def _next(self, deadline):
        """(arrival time, dict) for the next line before deadline; the dict is None for garbage, both on timeout/EOF."""
        try:
            at, line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except Empty:
            return None, None
        if line is None:
            self.ready = False
            return None, None
        try:
            msg = json.loads(line)
        except ValueError:
            return at, None
        return at, msg if isinstance(msg, dict) else None

    def send(self, obs):
        if self.ready is None:
            _, msg = self._next(self.ready_by)
            self.ready = bool(msg and msg.get("type") == "ready")
        self.sent = (obs["tick"], time.monotonic()) if self.ready else None
        if self.sent:
            self.out.put(obs)

    def recv(self, timeout=ANSWER_S):
        if self.sent is None:
            self.missed += 1
            return None
        tick, t0 = self.sent
        while True:
            at, act = self._next(t0 + timeout)
            if act is None or at is None:
                self.missed += 1
                return None
            if act.get("tick") == tick:
                self.answers += 1
                self.wait_s += at - t0
                self.late += at - t0 > LATE_S
                return act
            # An answer to an earlier tick that arrived after its timeout: skip it.

    def act(self, obs):
        self.send(obs)
        return self.recv()

    def close(self):
        LIVE.discard(self)
        self.out.put(None)
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=60)
        if self.proc.poll() is None:
            self.proc.kill()


@atexit.register
def _close_all():
    for b in list(LIVE):
        b.close()


def freeze_house():
    """Extract bot/ at HOUSE_REV once."""
    if (HOUSE_DIR / "brain.py").exists():
        return
    data = subprocess.run(["git", "archive", HOUSE_REV, "bot"], cwd=ROOT, capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(data)).extractall(HOUSE_DIR.parent, filter="data")


def load(ours_dir):
    """Import house on the frozen bot/, then our brain/params from ours_dir as a separate module set."""
    global house, ours, GENOME, CFG
    sys.path.insert(0, str(HERE))
    import engine  # puts ROOT/bot on sys.path and binds the simulator's world
    def purge(d):
        for p in Path(d).glob("*.py"):
            sys.modules.pop(p.stem, None)
    purge(HOUSE_DIR)
    sys.path.insert(0, str(HOUSE_DIR))
    import house
    frozen = sys.modules["brain"]
    sys.path.remove(str(HOUSE_DIR))
    purge(HOUSE_DIR)
    purge(ours_dir)
    sys.path.insert(0, str(Path(ours_dir).resolve()))
    import brain as ours
    import params
    assert house.Bot is frozen.Bot is not ours.Bot and Path(frozen.__file__) != Path(ours.__file__), \
        (frozen.__file__, ours.__file__)
    GENOME, CFG = params.load(), engine.load_config()


def factory(entry, images):
    if entry == "ours":
        return lambda h: ours.Bot(h, GENOME)
    return lambda h: DockerBot(images[entry], h)


def play(task):
    """One match. Returns a CSV row per seat of interest."""
    mode, seed, entries, images = task
    t0 = time.time()
    try:
        if mode == "practice":
            M, slot, _ = house.practice_match(CFG, seed, None, player=factory(entries[0], images),
                                              house=os.environ.get("LEAGUE_HOUSE", "v1"))
            seats = {slot: entries[0]}
        else:
            from engine import Match
            M = Match(CFG, seed, [factory(e, images) for e in entries])
            seats = dict(enumerate(entries))
        dealt, taken, died = defaultdict(float), defaultdict(float), {}
        damage, deaths = M._damage, M._deaths

        def tracked_damage(tgt, amt, att, *a, **k):
            hp = max(tgt.hp, 0.0)
            damage(tgt, amt, att, *a, **k)
            if tgt.kind == "mech":
                lost = hp - max(tgt.hp, 0.0)
                taken[tgt.id] += lost
                if att is not None and att.kind == "mech" and att is not tgt:
                    dealt[att.id] += lost

        def tracked_deaths():
            alive = [m for m in M.mechs if m.alive]
            deaths()
            for m in alive:
                if not m.alive:
                    died[m.id] = M.t
        M._damage, M._deaths = tracked_damage, tracked_deaths
        res = M.run()
    finally:
        _close_all()
    rows = []
    for s, entry in seats.items():
        m, b = M.mechs[s], M.mechs[s].bot
        cause = ""
        if not m.alive:
            k = getattr(m, "killer", None)
            last = [e for e in m.events if e["type"] == "damaged" and k is not None and e.get("by") == k.id]
            cause = "zone" if k is None else "self" if k is m else k.kind if k.kind != "mech" else \
                last[-1]["weapon"] if last else "mech"
        answers = getattr(b, "answers", 0)
        rows.append(dict(mode=mode, seed=seed, entry=entry, seat=s, points=16 - res["place"][s] if res["place"][s] else "",
                         place=res["place"][s] or "", kills=m.kills, dealt=round(dealt[s], 1), taken=round(taken[s], 1),
                         survived_s=round(died.get(s, res["t"]), 2), cause=cause, late=getattr(b, "late", 0),
                         missed=getattr(b, "missed", 0),
                         answer_ms=round(1000 * b.wait_s / answers, 1) if answers else "",
                         match_s=round(time.time() - t0, 2)))
    return rows


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else 0.0


def summary(rows, mode):
    by = defaultdict(list)
    for r in rows:
        by[r["entry"]].append(r)
    print(f"{'entry':<14}{'n':>5}{'points':>9}{'se':>7}{'kills':>7}{'surv_s':>8}{'late':>7}{'missed':>8}{'match_s':>9}")
    for e, rs in by.items():
        m, se = mean_se([float(r["points"]) for r in rs])
        print(f"{e:<14}{len(rs):>5}{m:>9.2f}{se:>7.2f}{sum(r['kills'] for r in rs) / len(rs):>7.2f}"
              f"{sum(r['survived_s'] for r in rs) / len(rs):>8.1f}{sum(r['late'] for r in rs):>7}"
              f"{sum(r['missed'] for r in rs):>8}{sum(r['match_s'] for r in rs) / len(rs):>9.1f}")
    names = list(by)
    if mode == "practice" and len(names) > 1:
        base = {r["seed"]: float(r["points"]) for r in by[names[0]]}
        for e in names[1:]:
            d = [float(r["points"]) - base[r["seed"]] for r in by[e] if r["seed"] in base]
            m, se = mean_se(d)
            print(f"{e} - {names[0]}: {m:+.2f} +/- {se:.2f} points over {len(d)} paired seeds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["practice", "ffa"])
    ap.add_argument("--entry", action="append", default=[], help="practice: entry to score (repeatable)")
    ap.add_argument("--field", default="", help="ffa: name:count,... (cycled to fill the seats)")
    ap.add_argument("--ours-seats", type=int, default=1)
    ap.add_argument("--ours", type=Path, default=ROOT / "bot", help="our bot/ directory")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--label", default="run")
    ap.add_argument("--house", choices=["v1", "v2"], default="v1", help="practice: house-bot model (sim/house.py)")
    args = ap.parse_args()
    roster = json.loads(ROSTER.read_text()) if ROSTER.exists() else {}
    seeds = range(args.seed_base, args.seed_base + args.seeds)
    if args.mode == "practice":
        if not args.entry:
            ap.error("practice needs at least one --entry")
        tasks = [(e, s) for s in seeds for e in args.entry]
        names = set(args.entry)
    else:
        field = [n for item in args.field.split(",") if item for n, c in [item.split(":")] for _ in range(int(c))]
        need = 16 - args.ours_seats
        if not field and need:
            ap.error("ffa needs --field")
        base = ["ours"] * args.ours_seats + [field[i % len(field)] for i in range(need)]
        tasks = []
        for s in seeds:
            seats = base[:]
            random.Random(s).shuffle(seats)
            tasks.append((seats, s))
        names = set(base)
    images = {n: roster.get(n, {}).get("image", f"league-{n}") for n in names - {"ours"}}
    for n, img in images.items():
        if subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode:
            sys.exit(f"entry {n}: docker image {img} not found")
    freeze_house()
    os.environ["LEAGUE_RUN"] = uuid.uuid4().hex[:6]  # container name prefix, inherited by the workers
    os.environ["LEAGUE_HOUSE"] = args.house
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.label}-{args.mode}.csv"
    new = not path.exists()
    rows = []
    t0 = time.time()
    pool = mp.get_context("spawn").Pool(args.jobs, initializer=load, initargs=(args.ours,), maxtasksperchild=50)
    try:
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            jobs = [(args.mode, s, [e] if args.mode == "practice" else e, images) for e, s in tasks]
            for out in pool.imap_unordered(play, jobs):
                w.writerows(out)
                f.flush()
                rows += out
                print(f"seed {out[0]['seed']} {out[0]['entry']}: {out[0]['match_s']} s", flush=True)
        pool.close()
        pool.join()
    finally:
        pool.terminate()
        # Workers killed mid-match skip their finally; sweep this run's containers.
        ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=league-{os.environ['LEAGUE_RUN']}-"],
                             capture_output=True, text=True).stdout.split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
    print(f"{len(tasks)} matches in {time.time() - t0:.0f} s wall, jobs={args.jobs}; rows in {path}")
    summary(rows, args.mode)


if __name__ == "__main__":
    main()

"""Repeatable aim and movement experiments using the match simulator and kit arena."""
import argparse
import csv
import gzip
import json
import math
import random
import subprocess
from collections import defaultdict
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from queue import Queue, Empty
from threading import Thread

from engine import DT, Match
from brain import Bot

ROOT = Path(__file__).resolve().parents[1]
WEAPONS = ("gun", "mortar", "laser", "lightning")


def ranges(weapon):
    return (15, 28, 40) if weapon == "lightning" else (30, 80, 150)


class Reference:
    def __init__(self, executable, hello, stack):
        self.process = subprocess.Popen([executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf-8")
        stack.callback(self.close)
        self.lines = Queue()
        Thread(target=self.read, daemon=True).start()
        if self.exchange(hello).get("type") != "ready":
            raise RuntimeError("Reference bot failed to start")

    def read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put("")

    def exchange(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        try:
            line = self.lines.get(timeout=15)
        except Empty as exc:
            raise RuntimeError("Reference bot timed out") from exc
        if not line:
            raise RuntimeError("Reference bot exited early")
        return json.loads(line)

    def act(self, obs):
        action = self.exchange(obs)
        if action.get("tick") != obs["tick"]:
            raise RuntimeError("Reference bot answered the wrong tick")
        return action

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


class AirTarget:
    """Cross the lane using normal game inputs, with repeatable jumps or jet bursts."""
    def __init__(self, mode):
        self.mode = mode
        self.center = None
        self.direction = 1
        self.next_jump = self.jet_until = 0

    def act(self, obs):
        me, t = obs["self"], obs["t"]
        y = me["pos"][1]
        if self.center is None:
            self.center = y
        if abs(y - self.center) >= 9:
            self.direction = -1 if y > self.center else 1
        jump = me["grounded"] and t >= self.next_jump
        if jump:
            self.next_jump = t + 2.0
            if self.mode == "jet" and me["energy"] >= 25:
                self.jet_until = t + 1.4
        return dict(tick=obs["tick"], move=[0, self.direction], jump=jump,
                    boost=t < self.jet_until, aim=[math.pi, 0])


class Trial(Match):
    """Observe the existing physics. Slot zero shoots and slot one is the target."""
    shots = hits = direct = 0
    damage = received = visible = 0.0
    first = None
    travel = air_time = jet_time = 0.0
    lightning_slot = None
    lightning_range_s = closing_damage = 0.0
    first_in_range = None

    def decide(self):
        super().decide()
        if self.errors:
            raise RuntimeError("Bot failed during the experiment")
        if not self.tick and not self.sees:
            raise RuntimeError("Invalid starting visibility")

    def step(self):
        self.visible += DT * self.sees
        target = self.mechs[1]
        start = (target.x, target.y)
        super().step()
        self.travel += math.dist(start, (target.x, target.y))
        self.air_time += DT * (not target.grounded)
        self.jet_time += DT * target.jetting

    def _weapons(self):
        if self.lightning_slot is not None:
            carrier, rival = self.mechs[self.lightning_slot], self.mechs[1-self.lightning_slot]
            dx, dy, dz = self._dir(carrier)
            muzzle = carrier.stats["muzzle_height"]
            barrel = self._barrel_offset(carrier, muzzle, dx, dy, dz)
            distance = math.dist((carrier.x + dx * barrel, carrier.y + dy * barrel, carrier.z + muzzle + dz * barrel),
                                 (rival.x, rival.y, rival.z + rival.h * 0.5))
            if distance <= carrier.wst["lightning"]["range"]:
                self.lightning_range_s += DT
                if self.first_in_range is None:
                    self.first_in_range = self.t
        before = len(self.sounds)
        super()._weapons()
        for kind, _, _, owner, _ in self.sounds[before:]:
            if kind in WEAPONS:
                if kind != self.weapons_by_slot[owner] or (owner == 1 and not self.duel):
                    raise RuntimeError(f"Unexpected discharge by bot {owner}: {kind}")
                self.shots += int(owner == 0)

    def _damage(self, tgt, amt, att, weapon, src, **kwargs):
        hp = tgt.hp
        super()._damage(tgt, amt, att, weapon, src, **kwargs)
        if self.lightning_slot is not None and self.first_in_range is None and tgt.hp < hp:
            if tgt is self.mechs[self.lightning_slot] and att is self.mechs[1-self.lightning_slot]:
                self.closing_damage += hp - max(0, tgt.hp)
        if tgt is self.mechs[0] and att is self.mechs[1] and tgt.hp < hp:
            self.received += hp - max(0, tgt.hp)
        if tgt is self.mechs[1] and att is self.mechs[0] and tgt.hp < hp:
            self.damage += hp - max(0, tgt.hp)
            if weapon != "mortar":
                self.hits += 1
            if self.first is None:
                self.first = self.t

    def _explode(self, p, ex, ey, ez, direct):
        previous = self.damage
        super()._explode(p, ex, ey, ez, direct)
        if self.damage > previous:
            self.hits += 1
            self.direct += int(direct is self.mechs[1])

    def _visibility(self, alive, actors):
        visibility = super()._visibility(alive, actors)
        self.sees = 1 in visibility[0]
        return visibility

    def _obs(self, m, *args):
        obs = super()._obs(m, *args)
        obs["alive"] = self.population
        return obs

    def _apply(self, m, action):
        if (self.aim and m.id == 0) or (self.frozen and m.id == 1):
            action = dict(action, move=[0, 0], boost=False, jump=False)
        if m.id == 1 and not self.duel:
            # The real pilot has already chosen this action. Preserve every other command.
            action = dict(action, fire=[False, False])
        elif self.weapons_by_slot[m.id] != "gun":
            action = dict(action, fire=[False, (action.get("fire") or [False, False])[1]])
        super()._apply(m, action)


@lru_cache(maxsize=None)
def lane(geometry, seed, distance, cover, radius, height):
    """Select a clear firing lane with a walkable sideways path at the target."""
    rng = random.Random(seed)
    for _ in range(20000):
        x, y = rng.uniform(-410, 320 - distance), rng.uniform(-300, 225)
        ground = geometry.ground
        eye = (x, y, ground(x, y) + height / 2)
        def clear(px, py):
            z = ground(px, py)
            return geometry.body_clear(px, py, z, radius, height) and geometry.surface(px, py) <= z + 0.2
        if not clear(x, y):
            continue
        tx = x + distance
        if not all(clear(tx, y + offset) for offset in range(-12, 13)):
            continue
        visible = []
        for offset in range(-12, 13):
            end = (tx, y + offset, ground(tx, y + offset) + height / 2)
            visible.append(geometry.sight(eye, end) and geometry.ray(eye, end) > 1)
        if not visible[12]:
            continue
        if cover:
            # A solid obstacle can block a shot from the initial firing position.
            if all(geometry.ray(eye, (tx, y + d, ground(tx, y + d) + height / 2)) > 1
                   for d in (-12, -8, 8, 12)):
                continue
        elif not all(visible):
            continue
        return x, y
    raise RuntimeError(f"No suitable lane for {distance} m, seed {seed}, cover={cover}")


def trial(cfg, arena, executable, seed, weapon, distance, setup, population, movement, gallery=None,
          *, opponent_weapon=None):
    aim = setup == "aim"
    if opponent_weapon is not None and (setup != "duel" or opponent_weapon not in WEAPONS):
        raise ValueError("opponent_weapon requires a duel and a valid weapon")
    with ExitStack() as stack:
        def reference(hello):
            return Reference(executable, hello, stack)
        if aim:
            factories = [Bot, reference if movement == "live" else lambda hello: AirTarget(movement)]
        elif setup == "duel":
            factories = [Bot, reference]
        else:
            factories = [reference, Bot]
        game = Trial(cfg, seed, factories, map_dir=str(arena))
        game.aim, game.frozen, game.population = aim, movement == "frozen", population
        game.duel = setup == "duel"
        game.weapons_by_slot = (weapon, opponent_weapon or weapon)
        if game.duel and game.weapons_by_slot.count("lightning") == 1:
            game.lightning_slot = game.weapons_by_slot.index("lightning")
        shooter, defender = game.mechs
        x, y = lane(game.G, seed, distance, setup == "cover", shooter.r, shooter.h)
        for m, px, yaw in [(shooter, x, 0), (defender, x + distance, math.pi)]:
            m.x, m.y, m.z = px, y, game.G.ground(px, y)
            m.yaw = m.cyaw = yaw
        armed_pilots = (shooter, defender) if game.duel else (shooter,)
        for pilot in armed_pilots:
            kind = game.weapons_by_slot[pilot.id]
            if kind != "gun":
                pilot.slots[1] = dict(pilot.slots[0], kind=kind)
                if not aim:
                    pilot.slots[0] = None
        limit = 15 if aim else 60
        if gallery:
            from viewer import record
            slot = 0 if aim or game.duel else 1
            recording = record(game, every=2, slot=slot, end_time=limit)
        else:
            while game.t < limit and shooter.alive and defender.alive:
                if game.tick % 2 == 0:
                    game.decide()
                game.step()
        assert game.hits <= game.shots
        row = dict(setup=setup, seed=seed, weapon=weapon, distance=distance, movement=movement,
                    observed_alive=population, x=round(x, 3), y=round(y, 3), seconds=round(game.t, 2),
                    shooter_alive=shooter.alive, target_alive=defender.alive,
                    shooter_damage_taken=round(game.received, 2),
                    survived_limit=game.t >= limit and defender.alive, shots=game.shots, hits=game.hits,
                    hit_pct=round(100 * game.hits / game.shots, 2) if game.shots else None,
                    target_travel_m=round(game.travel, 2), target_air_s=round(game.air_time, 2),
                    target_jet_s=round(game.jet_time, 2), direct_mortar=game.direct, damage=round(game.damage, 2),
                    damage_per_s=round(game.damage / game.t, 2), visible_s=round(game.visible, 2),
                    first_hit_s=game.first if game.first is None else round(game.first, 2),
                    in_flight=sum(p.owner is shooter for p in game.projs))
        if opponent_weapon is not None:
            row.update(opponent_weapon=opponent_weapon, lightning_range_s=round(game.lightning_range_s, 2),
                       first_in_range_s=round(game.first_in_range, 2) if game.first_in_range is not None else None,
                       lightning_closing_damage=round(game.closing_damage, 2) if game.lightning_slot is not None else None)
        if gallery:
            gallery.save(row, recording, slot, cfg)
        return row


def print_report(rows):
    aim = defaultdict(list)
    dodge = defaultdict(dict)
    for row in rows:
        if row["setup"] == "aim":
            aim[row["weapon"], row["distance"], row["movement"]].append(row)
        elif row["setup"] != "duel":
            key = row["observed_alive"], row["setup"], row["weapon"], row["distance"], row["seed"]
            dodge[key][row["movement"]] = row
    print("Aim: weapon, range, target, hits/shots, damage/s, trials without shots")
    for key, group in aim.items():
        hits = sum(r["hits"] for r in group)
        shots = sum(r["shots"] for r in group)
        dps = sum(r["damage"] for r in group) / sum(r["seconds"] for r in group)
        print(*key, f"{hits}/{shots}", f"{dps:.1f}", sum(r["shots"] == 0 for r in group))
    print("Dodge: observed alive, setup, weapon, distance, seed, frozen seconds, normal seconds, change")
    for key, pair in dodge.items():
        frozen, normal = pair["frozen"], pair["normal"]
        delta = normal["seconds"] - frozen["seconds"]
        warning = " (stationary control took no hits)" if not frozen["hits"] else ""
        print(*key, frozen["seconds"], normal["seconds"], f"{delta:+.2f}{warning}")


    print("Two-way duels: weapon, distance, seed, seconds, our health outcome, their health outcome")
    for row in rows:
        if row["setup"] == "duel":
            print(row["weapon"], row["distance"], row["seed"], row["seconds"],
                  "alive" if row["shooter_alive"] else "dead", "alive" if row["target_alive"] else "dead")


def cases_for_seed():
    for population in (2, 16):
        for weapon in WEAPONS:
            for distance in ranges(weapon):
                for movement in ("live", "jump", "jet"):
                    yield weapon, distance, "aim", population, movement
                for setup in ("open", "cover"):
                    for movement in ("normal", "frozen"):
                        yield weapon, distance, setup, population, movement
    for weapon in WEAPONS:
        for distance in ranges(weapon):
            yield weapon, distance, "duel", 2, "both"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=ROOT / "kit")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, help="Write the interactive case gallery here")
    parser.add_argument("--weapon-swap", action="store_true",
                        help="Only run gun versus lightning, then swap weapons on the same 65 m lane")
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    args = parser.parse_args()
    kit = args.kit.resolve()
    with gzip.open(kit / "check/stream.jsonl.gz", "rt") as stream:
        cfg = json.loads(stream.readline())["config"]
    cfg["match"]["mechs"] = 2
    cfg["npc"]["count"] = cfg["crates"]["count"] = cfg["loot"]["floor_count"] = 0
    cfg["drops"]["times"] = []
    # Fixed safe area isolates weapon damage and removes zone timing as a variable.
    for phase in cfg["zone"]["phases"]:
        phase["hold_s"] = 120
    gallery = None
    if args.gallery:
        from combat_gallery import Gallery
        gallery = Gallery(args.gallery, kit / "arena")
    rows = []
    for seed in args.seeds:
        if args.weapon_swap:
            for weapon, opponent in (("gun", "lightning"), ("lightning", "gun")):
                rows.append(trial(cfg, kit / "arena", args.reference.resolve(), seed, weapon, 65, "duel", 2,
                                  "both", gallery, opponent_weapon=opponent))
        else:
            for weapon, distance, setup, population, movement in cases_for_seed():
                rows.append(trial(cfg, kit / "arena", args.reference.resolve(), seed, weapon, distance,
                                  setup, population, movement, gallery))
        print(f"Finished seed {seed}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if gallery:
        gallery.finish(rows, args.output)
    print_report(rows)


if __name__ == "__main__":
    main()

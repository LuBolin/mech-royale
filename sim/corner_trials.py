"""Mirrored three-mech corners and fresh-attacker scenarios. Run once per bot checkout."""
import argparse
import csv
import math
from pathlib import Path
import sys

from engine import DT, Match, load_config, wrap

ROOT = Path(__file__).resolve().parents[1]
# Opposite-handed corners of real buildings. Local offsets are identical. The
# wall tangent points from the corner into cover, and the normal into the wall.
CORNERS = {
    "left": ((-1.4, -189.9), (0.154486, -0.987995), (0.987995, 0.154486)),
    "right": ((-61.7, -223.3), (-0.155393, 0.987853), (0.987853, 0.155393)),
}


class Guard:
    """Hold this lane. Return fire for two seconds after the third mech hits us."""
    def __init__(self, hello, focus):
        self.priority = 3 - hello["slot"] if focus == "each_other" else 0
        self.respond_until = -1

    def act(self, obs):
        me = obs["self"]
        if any(e["type"] == "damaged" and e.get("by") == 0 for e in obs["events"]):
            self.respond_until = obs["t"] + 2
        wanted = 0 if obs["t"] < self.respond_until else self.priority
        enemies = {m["id"]: m for m in obs["mechs"] if m["kind"] == "mech"}
        target = enemies.get(wanted)
        if target is None and obs["alive"] == 2 and enemies:
            target = next(iter(enemies.values()))
        if target is None:
            return {"tick": obs["tick"], "aim": me["aim"]}
        x, y, z = me["pos"]
        tx, ty, tz = target["pos"]
        vx, vy, vz = target["vel"]
        flight = math.hypot(tx-x, ty-y) / me["weapons"][0]["stats"]["speed"]
        dx, dy = tx + vx*flight - x, ty + vy*flight - y
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(tz + 4.5 + vz*flight - z - me["stats"]["muzzle_height"], math.hypot(dx, dy))
        aligned = abs(wrap(yaw-me["aim"][0])) < 0.15 and abs(pitch-me["aim"][1]) < 0.15
        return {"tick": obs["tick"], "aim": [yaw, pitch], "fire": [aligned, False]}


class CornerMatch(Match):
    dealt = taken = double_s = 0.0

    def _damage(self, target, amt, attacker, weapon, src, **kwargs):
        hp = max(target.hp, 0.0)
        super()._damage(target, amt, attacker, weapon, src, **kwargs)
        lost = hp - max(target.hp, 0.0)
        if attacker is self.mechs[0] and target is not attacker:
            self.dealt += lost
        if target is self.mechs[0] and attacker in self.mechs[1:]:
            self.taken += lost

    def exposed(self, pos):
        return sum(m.alive and math.dist((m.x, m.y), pos[:2]) < m.wst["gun"]["range"]
                   and self.G.ray((m.x, m.y, m.z+m.stats["muzzle_height"]), pos) > 1
                   for m in self.mechs[1:])


def run(bot, corner, focus, limit, hold=False):
    cfg = load_config()
    cfg["match"]["mechs"] = 3
    cfg["npc"]["count"] = cfg["loot"]["floor_count"] = cfg["crates"]["count"] = 0
    cfg["drops"]["times"] = []
    for phase in cfg["zone"]["phases"]:
        phase["hold_s"] = limit + 1
    origin, tangent, normal = CORNERS[corner]
    class HoldCover:
        def act(self, obs):
            x, y, _ = obs["self"]["pos"]
            dx, dy = origin[0]+12*tangent[0]-6*normal[0]-x, origin[1]+12*tangent[1]-6*normal[1]-y
            distance = max(2, math.hypot(dx, dy))
            return dict(tick=obs["tick"], move=[dx/distance, dy/distance])
    guard = lambda hello: Guard(hello, focus)
    game = CornerMatch(cfg, 517, [lambda hello: HoldCover() if hold else bot(hello), guard, guard])
    def point(a, b):
        x = origin[0] + a*tangent[0] + b*normal[0]
        y = origin[1] + a*tangent[1] + b*normal[1]
        return x, y, game.G.ground(x, y)
    our, first, second = game.mechs
    for m, position in zip(game.mechs, (point(-4, -6), point(-12, 32), point(-40, 26))):
        m.x, m.y, m.z = position
        assert game.G.body_clear(*position, m.r, m.h)
    for m, opponent in zip(game.mechs, (first, second if focus == "each_other" else our,
                                        first if focus == "each_other" else our)):
        m.yaw = m.cyaw = math.atan2(opponent.y-m.y, opponent.x-m.x)
    # Validate that the actual firing lanes progress from none to one to two as
    # the mech peeks. Start wide so both guns can engage immediately.
    exposures = []
    for offset in range(12, -5, -1):
        x, y, z = point(offset, -6)
        exposures.append(game.exposed((x, y, z+our.h/2)))
    assert exposures == sorted(exposures) and set(exposures) == {0, 1, 2}
    # A wounded duelist makes a third-party finish possible. Use identical HP
    # for both bot revisions and both corners.
    if focus == "each_other":
        first.hp *= 0.55
    join = focus.startswith("join_")
    if join:
        # The second shooter enters the same lane and immediately focuses us.
        # Keep it outside sensor and weapon range until the scripted arrival.
        join_position = (second.x, second.y, second.z)
        second.x, second.y = (400, 100)
        second.z = game.G.surface(second.x, second.y)
    healed = 0
    while game.t < limit - 1e-8 and our.alive and (first.alive or second.alive):
        if join and game.tick == 20:
            second.x, second.y, second.z = join_position
            second.yaw = second.cyaw = math.atan2(our.y-second.y, our.x-second.x)
            first.hp = 70 if focus == "join_finish" else first.max_hp
        if game.tick % 2 == 0:
            game.decide()
            if game.errors:
                raise RuntimeError("Bot failed during the corner trial")
        game.double_s += DT * (game.exposed((our.x, our.y, our.z+our.h/2)) == 2)
        previous_hp = our.hp
        game.step()
        healed += max(0, our.hp-previous_hp)
    place = our.place if not our.alive else (1 if not first.alive and not second.alive else None)
    return dict(corner=corner, focus="hold" if hold else focus, place=place,
                others_alive=int(first.alive)+int(second.alive), alive=our.alive, seconds=round(game.t, 2),
                double_exposure_s=round(game.double_s, 2), damage_dealt=round(game.dealt),
                damage_taken=round(game.taken), recovered_hp=round(healed),
                original_down=not first.alive, kills=our.kills)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot-dir", type=Path, default=ROOT / "bot", help="bot/ directory from the revision to evaluate")
    parser.add_argument("--seconds", type=float, default=35)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hold-control", action="store_true", help="Also run a passive retreat on the two final-three corners")
    args = parser.parse_args()
    # Match imported its simulator world above. Bind the pilot's own modules,
    # genome and baked map from the requested checkout in this fresh process.
    sys.modules.pop("world", None)
    sys.path.insert(0, str(args.bot_dir.resolve()))
    from brain import Bot
    rows = [run(Bot, corner, focus, args.seconds) for corner in CORNERS for focus in ("us", "each_other")]
    rows.extend(run(Bot, "left", focus, args.seconds) for focus in ("join_finish", "join_healthy"))
    if args.hold_control:
        rows.extend(run(Bot, corner, "each_other", args.seconds, hold=True) for corner in CORNERS)
    def write(stream):
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write(sys.stdout)
    if args.output:
        with args.output.open("w", newline="") as stream:
            write(stream)


if __name__ == "__main__":
    main()
